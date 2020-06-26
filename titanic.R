# This file includes R code to perform data analysis on the 
# Popular Titanic ship life survival data

# The dataset is already available as an inbuilt dataset in R
# The goal of the analysis is to analyze the various predictors
# and predict who will Survive

# The analysis on this dataset was done as part of the course work in 
# HarvardX PH125.9x Certificate Program


library(titanic)    # loads titanic_train data frame
library(caret)
library(tidyverse)
library(rpart)

# 3 significant digits
options(digits = 3)

# clean the data - `titanic_train` is loaded with the titanic package
# factorize character vectors and fill missing data for Age with median values
titanic_clean <- titanic_train %>%
  mutate(Survived = factor(Survived),
         Embarked = factor(Embarked),
         Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age), # NA age to median age
         FamilySize = SibSp + Parch + 1) %>%    # count family members
  select(Survived,  Sex, Pclass, Age, Fare, SibSp, Parch, FamilySize, Embarked)

head(titanic_clean)

# Survived column is the outcome
y<- titanic_clean$Survived

# create train and test datasets
set.seed(42, sample.kind="Rounding") #if using R 3.6 or later
test_index <- createDataPartition(y, times = 1, p = 0.2, list = FALSE)
train_set <- titanic_clean %>% slice(-test_index) 
test_set <- titanic_clean %>% slice(test_index)

# Percentage of Survival
mean(train_set$Survived==1)


# The first predictive model to evaluate is the guessing model
# based on different predictors

set.seed(3, sample.kind="Rounding") #if using R 3.6 or later
guess<-sample(c(0,1), length(test_set$Survived), replace=TRUE)%>%
  factor(levels=levels(test_set$Survived))

results<-data.frame(method="random guess", accuracy=mean(guess==test_set$Survived))
results
##########################

table(titanic_clean$Survived, titanic_clean$Sex)

# Survival of Male & Female showed that only 20% of male survived 
# but 73% of female survived.
mean(train_set$Survived[train_set$Sex=="female"]==1)
mean(train_set$Survived[train_set$Sex=="male"]==1)


# Improve the Guess model by predicting based on gender
guess_3b<-ifelse(test_set$Sex=="female", 1,0)%>%
  factor(levels=levels(test_set$Survived))

results<-rbind(results, data.frame(method="female guess", accuracy=mean(guess_3b==test_set$Survived)))
results

##########################
# Improve model by including PClass 

table(titanic_clean$Survived, titanic_clean$Pclass)

train_set%>%
  group_by(Pclass)%>%
  summarize(sum=sum(Survived==1), mean=mean(Survived==1))
# Summary shows that 1st class has more survivals


guess_4b<-ifelse(test_set$Pclass==1, 1,0)%>%
  factor(levels=levels(test_set$Survived))

results<-rbind(results, data.frame(method="Pclass guess", accuracy=mean(guess_4b==test_set$Survived)))
results

# The results summary shows that the accuracy decreased from female guess model 
# but it is already known that Pclass=1 has better survival

##########################
# Improve model by including PClass and Sex

train_set%>%
  group_by(Pclass, Sex)%>%
  summarize(sum=sum(Survived==1), mean=mean(Survived==1))

guess_4d<-ifelse(test_set$Sex=="female" & (test_set$Pclass==1|test_set$Pclass==2), 1, 0) %>% 
  factor(levels=levels(test_set$Survived))

results<-rbind(results, data.frame(method="Pclass & Sex guess", accuracy=mean(guess_4d==test_set$Survived)))
results

# Accuracy drop down to 65%. Guess model with female model showed best results

confusionMatrix(guess, test_set$Survived)
confusionMatrix(guess_3b, test_set$Survived)
confusionMatrix(guess_4b, test_set$Survived)
confusionMatrix(guess_4d, test_set$Survived)

F_meas(guess_3b, test_set$Survived)
F_meas(guess_4b, test_set$Survived)
F_meas(guess_4d, test_set$Survived)

##############################################################
# Evaluating Machine Learning Models 
# Each of these models were run and accuracy compared

# LDA model

set.seed(1, sample.kind="Rounding") #if using R 3.6 or later
train_lda<-train(Survived~., method="lda", data=train_set)
yhat_lda<-predict(train_lda, test_set)

results<-rbind(results, data.frame(method="LDA", accuracy=mean(yhat_lda==test_set$Survived)))
results


# QDA model

set.seed(1, sample.kind="Rounding") #if using R 3.6 or later
train_qda<-train(Survived~Sex+Age+Pclass+Fare, method="qda", data=train_set)
yhat_qda<-predict(train_qda, test_set)

results<-rbind(results, data.frame(method="QDA", accuracy=mean(yhat_qda==test_set$Survived)))
results


# Logistic Regression

set.seed(1, sample.kind="Rounding") #if using R 3.6 or later
train_glm<-train(Survived~., method="glm", data=train_set)#%>%factor(levels=levels(train_set$Survived))
yhat_glm<-predict(train_glm, test_set)

results<-rbind(results, data.frame(method="GLM", accuracy=mean(yhat_glm==test_set$Survived)))
results


# KNN Model with cross validation

set.seed(6, sample.kind="Rounding") #if using R 3.6 or later
control<-trainControl(method="cv", number=10, p=0.9)
train_knn<-train(Survived~., 
                 method="knn", 
                 tuneGrid=data.frame(k=seq(3,51,2)), 
                 data=train_set, 
                 trControl=control)
ggplot(train_knn)
train_knn$bestTune
yhat_knn<-predict(train_knn, test_set)

results<-rbind(results, data.frame(method="KNN", accuracy=mean(yhat_knn==test_set$Survived)))
results


# CART Model with fine tuning and cross validation

set.seed(10, sample.kind="Rounding") #if using R 3.6 or later
cp<-seq(0,0.05,0.002)
control<-trainControl(method="cv", number=10, p=0.9)
train_rpart<-train(Survived~., 
                   method="rpart", 
                   tuneGrid=data.frame(cp=cp), 
                   data=train_set,
                   trControl=control)
ggplot(train_rpart)
train_rpart$bestTune
yhat_rpart<-predict(train_rpart, test_set)

train_rpart$finalModel
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel)

results<-rbind(results, data.frame(method="CART", accuracy=mean(yhat_rpart==test_set$Survived)))
results


# Random Forest Model with fine tuning and cross validation

set.seed(14, sample.kind="Rounding") #if using R 3.6 or later
train_rf<-train(Survived~., 
                method="rf", 
                tuneGrid = data.frame(mtry = seq(1,7,1)), 
                data=train_set, 
                ntree=100,
                trControl=control)
ggplot(train_rf)
train_rf$bestTune
yhat_rf<-predict(train_rf, test_set)

results<-rbind(results, data.frame(method="RF", accuracy=mean(yhat_rf==test_set$Survived)))
results

# Variable importance table
varImp(train_rf)

# final results summary shows that the Logistic Regression Model showed best accuracy
# However, Random Forest, CART and LDA are very close too.



