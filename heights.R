# The height datasets were collected from three classes taught in the
# Departments of Computer Science and Biostatistics, as well as remotely 
# through the Extension School. The biostatistics class was taught in 2016  
# along with an online version offered by the Extension School.

# The analysis on this dataset was done as part of the course work in 
# HarvardX PH125.9x Certificate Program

# This dataset includes the sex of the people and their heights.
# The goal of the analysis is to predict the sex of a person based on height

library(tidyverse)
library(caret)
library(dslabs)
data(heights)

head(heights)

# define predictor and outcome
y <- heights$sex
x <- heights$height

# Data exploration reveals that Female proportion in the dataset 
# is much lower than Male proportion
heights%>%
  group_by(sex)%>%
  summarize(n=n(), .groups='drop')%>%
  ggplot(aes(sex, n, fill=sex))+
  geom_bar(stat="identity")

# heights are almost in normal distribution however, males and females have 
# different means and stdevs
heights%>%
  group_by(ht=round(height, 0), sex)%>%
  summarize(n=n(), .groups='drop')%>%
  ggplot(aes(ht, n, fill=sex))+
  geom_bar(stat="identity", position = "dodge")


#Split data for train and test sets
set.seed(2007, sample.kind="Rounding")
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
test_set <- heights[test_index, ]
train_set <- heights[-test_index, ]

################################################
# First model to randomly guess the sex
y_hat <- sample(c("Male", "Female"), length(test_index), replace = TRUE) %>%
  factor(levels = levels(test_set$sex))

# Accuracy of random guess
guess_acc<-mean(y_hat == test_set$sex)

results<-data.frame(method="Random Guess", Accuracy=guess_acc)


################################################
# Compute mean and stdev of the heights
heights %>% 
  group_by(sex) %>% 
  summarize(mean(height), sd(height))

# predict Male if height is more than 2 stdev above mean i.e. ht>62
y_hat <- ifelse(x > 62, "Male", "Female") %>% 
  factor(levels = levels(test_set$sex))

# Accuracy of Guessing Male height above mean-2*stdev
guess_acc_stdev<-mean(y_hat == test_set$sex)

results<-rbind(results, data.frame(method="2 Stdev Above Mean", Accuracy=guess_acc_stdev))

################################################
# guess Male with higher probablity as dataset is imbalanced 
# with more males than females

p <- 0.9
n <- length(test_index)
y_hat <- sample(c("Male", "Female"), n, replace = TRUE, prob=c(p, 1-p)) %>% 
  factor(levels = levels(test_set$sex))

guess_acc_prob<-mean(y_hat == test_set$sex)

results<-rbind(results, data.frame(method="Probability Guess", Accuracy=guess_acc_prob))

################################################
# Guessing with cutoff Analysis

# This method sweeps different height values as cutoff to label Male
# User defined function calculates accuracy to pick cutoff value
# for maximum accuracy


# Sweep cutoff heights from 61 to 70 and caclulate accuracy
cutoff <- seq(61, 70)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train_set$height > x, "Male", "Female") %>% 
    factor(levels = levels(test_set$sex))
  mean(y_hat == train_set$sex)
})

#plot cutoff vs accuracy
plot(cutoff, accuracy, main="Height Cutoff Versus Accuracy")

#Pick max accuracy
max(accuracy)

#choose cutoff for max accuracy
acc_cutoff <- cutoff[which.max(accuracy)]
acc_cutoff

# Label using the cutoff value obtained above
y_hat <- ifelse(test_set$height > acc_cutoff, "Male", "Female") %>% 
  factor(levels = levels(test_set$sex))
y_hat <- factor(y_hat)

# Accuracy on testset with the cutoff value
cutoff_acc<-mean(y_hat == test_set$sex)

results<-rbind(results, data.frame(method="Accuracy-Cutoff Analysis ", Accuracy=cutoff_acc))

# Confusion Matrix
table(predicted = y_hat, actual = test_set$sex)

# The confusion matrix shows that the female accuracy is very low
# even though the overall accuracy has improved. This is because
# the dataset is imbalanced or the prevelance of female is low
# Sensitivity and Specificity are other measures that can give 
# more information on model prediction
cm <- confusionMatrix(data = y_hat, reference = test_set$sex)
cm$overall["Accuracy"]
cm$byClass[c("Sensitivity","Specificity", "Prevalence")]


# Rebuild algorithm to maximize Fscore instead of accuracy
cutoff <- seq(61, 70)
F_1 <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train_set$height > x, "Male", "Female") %>% 
    factor(levels = levels(test_set$sex))
  F_meas(data = y_hat, reference = factor(train_set$sex))
})

#plot F1 vs cutoff
plot(cutoff, F_1, main="Height Cutoff Versus F Score")

#Maximum F1
max(F_1)

#Pick cutoff for maximum FScore
F1_cutoff <- cutoff[which.max(F_1)]
F1_cutoff
# Turns out that the cutoff value for max accuracy and max F1 is the same

# Label using the new cutoff obtained from FScore maximizing
y_hat <- ifelse(test_set$height > F1_cutoff, "Male", "Female") %>% 
  factor(levels = levels(test_set$sex))
y_hat <- factor(y_hat)

# Accuracy on testset with the cutoff value
cutoff_F1<-mean(y_hat == test_set$sex)

results<-rbind(results, data.frame(method="F1 Score-Cutoff Analysis ", Accuracy=cutoff_F1))

#Confusion Matrix
cm <- confusionMatrix(data = y_hat, reference = test_set$sex)
cm$overall["Accuracy"]
cm$byClass[c("Sensitivity","Specificity", "Prevalence")]


################################################
# KNN Model


k<- seq(1, 101, 3)
result <- sapply(k, function(k) {
  knn_fit<-knn3(sex ~ ., data=train_set, k=k)
  y_hat<-predict(knn_fit, test_set, type="class")%>%
    factor(levels=levels(test_set$sex))
  F_meas(data=y_hat, reference=factor(test_set$sex))
})

# Maximum F Score
max(result)
k[which.max(result)]

# The FScore from KNN Model has increased to 0.6216 from the 
# FScore of the height cutoff analysis at 0.5887

# By using more Machine Learning algorithms the F Score can be 
# improved further

