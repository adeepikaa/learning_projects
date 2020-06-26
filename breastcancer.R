########################################################################################
#                                   INTRODUCTION
########################################################################################

# The Breast Cancer Wisconsin (Diagnostic Dataset) contains features computed from a
# digitized image of a fine needle aspirate (FNA) of a breast mass.  These features 
# describe the characteristics of the cell nuclei present in the image.

# The goal is to process these features and predict of the mass is Benign or Malignant.

# The dslabs library includes the BreastCancer dataset which has all the predictors (x)
# and y (outcome) in matrix format
# The predictors are based on features that are further described below

# The analysis on this dataset was done as part of the course work in 
# HarvardX PH125.9x Certificate Program


# Load libraries
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)
data(brca)

# Dimensions of the features
dim(brca$x)

length(brca$y)

# 3 significant digits
options(digits = 3)

str(brca$x)

# Attribute Information:
#   
#   1) ID number
#   2) Diagnosis (M = malignant, B = benign)
# 3-32) Features :
# 
# Ten real-valued features are computed for each cell nucleus:
#   
# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry
# j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest 
# (mean of the three largest values) of these features were computed for each image,
# resulting in 30 features. 

# Class distribution: 357 benign, 212 malignant

sum(brca$y=="M")
sum(brca$y=="B")

mean(brca$y=="M")

# Which column has max mean 
which.max(colMeans(brca$x))

# Which column has min std dev
which.min(colSds(brca$x))

# Use scaling on the features 
brca$x_sc <- sweep(brca$x, 2, colMeans(brca$x, na.rm=TRUE), "-")
brca$x_sc <- sweep(brca$x_sc, 2, colSds(brca$x, na.rm=TRUE), "/")

sd(brca$x_sc[,1])
median(brca$x_sc[,1])

############################################
#  Feature exploration and understanding
############################################

# Calculate distances between samples
d<-as.matrix(dist(brca$x_sc))
index<-which(brca$y=="B")
mean(d[index,1])
index<-which(brca$y=="M")
mean(d[index,1])


d_samples <- dist(brca$x_sc)
dist_BtoB <- as.matrix(d_samples)[1, brca$y == "B"]
mean(dist_BtoB[2:length(dist_BtoB)])
dist_BtoM <- as.matrix(d_samples)[1, brca$y == "M"]
mean(dist_BtoM)

heatmap(as.matrix(d), labRow = NA, labCol = NA)

# Heirarchical clustering on the features
d_features <- dist(t(brca$x_sc))
heatmap(as.matrix(d_features), labRow = NA, labCol = NA)
h <- hclust(d_features)
groups <- cutree(h, k = 5)
split(names(groups), groups)

# Clustering shows which features are in the same clusters

###########################################
#  Principal Component Analysis 
# (used to understand the importance of features)

#Run PCA
pca<-prcomp(brca$x_sc)

# Summary of PCA yields that the first 7 Principal components 
# contribute to about 91% of the variation
summary(pca)

# Malignant tumors tend to have higher PC1 values than PC2
data.frame(pca$x[,1:2], Species=brca$y) %>% 
  ggplot(aes(PC1,PC2, fill = Species))+
  geom_point(cex=3, pch=21) +
  coord_fixed(ratio = 1)


# Box plots of first 10 PCs
# Most PCs have overlapping inter-quartile ranges except PC1
for(i in 1:10){
  boxplot(pca$x[,i] ~ brca$y, main = paste("PC", i))
}


# All PCs in 1 plot show that PC2 has wider range like PC1 
# with median values that are not close like other PCs
data.frame(type = brca$y, pca$x[,1:10]) %>%
  gather(key = "PC", value = "value", -type) %>%
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot()




############################################
#  Predictive Analysis
############################################

# using the scaled values for predictive algorithms
x_scaled<-brca$x_sc
set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later

# creating train and test datasets
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- brca$y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- brca$y[-test_index]

# checking the distribution of Malignancy
mean(train_y=="B")
mean(test_y=="B")
# Very similar 62%


#############################################

# K-Means Clustering

# Run Kmeans with two clusters
set.seed(3, sample.kind = "Rounding")    # if using R 3.6 or later
k <- kmeans(train_x, centers = 2)


predict_kmeans <- function(x, k) {

  centers <- k$centers    # extract cluster centers of Kmeans model
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y))) 
    # get distance between the i th x for all cols and y(center of cluster)
  })
  # select cluster with min distance to center
  # transpose the distances and pick the column name with min distances
  max.col(-t(distances))  # - sign to negate for max to get min
}

# Based on Cluster selected from above, assign B or M labels
kmeans_predict<-ifelse(predict_kmeans(test_x, k)==1, "B","M")

# predicted Accuracy of all predictions 92.2%
kmeans_acc<-mean(kmeans_predict==test_y)

# Confusion Matrix 
table(kmeans_predict, test_y)

# TPR true positive rate of Cancer
tpr<-sum(kmeans_predict=="B" & test_y== "B")/sum(test_y=="B")

# TNR true negative rate of Cancer
tnr<-sum(kmeans_predict=="M" & test_y== "M")/sum(test_y=="M")

results<-data.frame(method="Kmeans clustering", accuracy=kmeans_acc)
results%>%knitr::kable()


#############################################

# Logistic Regression

#model using caret package
model_glm<-train(train_x, train_y, method = "glm")

#predict outcomes for test set
glm_y<-  predict(model_glm, test_x)

# calculate accuracy
mean(glm_y==test_y)

# add to results to summarize results across different models
results<-rbind(results, data.frame(method="Logistic Regression", accuracy=mean(glm_y==test_y)))
results%>%knitr::kable()


#############################################

# Linear Discriminant Analysis

#model using caret package
model_lda<-train(train_x, train_y, method = "lda")

#predict outcomes for test set
lda_y<-  predict(model_lda, test_x)

# calculate accuracy
mean(lda_y==test_y)

# add to results to summarize results across different models
results<-rbind(results, data.frame(method="Linear Discriminant Analysis", accuracy=mean(lda_y==test_y)))
results%>%knitr::kable()


#############################################

# Quadratic Discriminant Analysis

#model using caret package
model_qda<-train(train_x, train_y, method = "qda")

#predict outcomes for test set
qda_y<-  predict(model_qda, test_x)

# calculate accuracy
mean(qda_y==test_y)

# add to results to summarize results across different models
results<-rbind(results, data.frame(method="QuadraticDiscriminant Analysis", accuracy=mean(qda_y==test_y)))
results%>%knitr::kable()


#############################################

# Gam Loess

#model using caret package
model_loess<-train(train_x, train_y, method = "gamLoess")

#predict outcomes for test set
loess_y<-  predict(model_loess, test_x)

# calculate accuracy
mean(loess_y==test_y)

# add to results to summarize results across different models
results<-rbind(results, data.frame(method="GAM Loess", accuracy=mean(loess_y==test_y)))
results%>%knitr::kable()


#############################################

# KNN -K Nearest Neighbors


set.seed(7, sample.kind = "Rounding")    # if using R 3.6 or later

#model using caret package, fine tune to find optimum k
train_knn <- train(train_x, train_y, method = "knn", 
                   tuneGrid = data.frame(k = seq(3, 51, 2)))

# accuracy Vs k plot
ggplot(train_knn, highlight = TRUE)

# final Model picks the best accuracy model among all models for different k values
train_knn$finalModel

# Best k value
train_knn$bestTune

#predict outcomes for test set
knn_preds <- predict(train_knn, test_x)

# calculate accuracy
mean(knn_preds == test_y)

# add to results to summarize results across different models
results<-rbind(results, data.frame(method="KNN", accuracy=mean(knn_preds==test_y)))
results%>%knitr::kable()



#############################################

# Random Forest Model


# model using caret package fien tuning for optimum mtry value
# mtry is number of variables at each node
set.seed(9, sample.kind = "Rounding")  
model_rf <- train(train_x, train_y, method = "rf", nodesize = 1,importance=TRUE,
                  tuneGrid = data.frame(mtry = c(3,5,7,9)))
model_rf$bestTune

#predict outcomes for test set
rf_y<-  predict(model_rf, test_x)

# calculate accuracy
mean(rf_y==test_y)

# Variable Importance plot shows the most contributing features
imp<-varImp(model_rf)

imp%>%
  ggplot(aes(names(), Importance, fill=names()))+geom_bar(stat="identity")

# add to results to summarize results across different models
results<-rbind(results, data.frame(method="Random Forest", accuracy=mean(rf_y==test_y)))
results%>%knitr::kable()


#############################################

# Ensemble Model by voting method

# Collect all predicted outcomes from different models
y_hat<-cbind(glm_y, lda_y, knn_preds, loess_y, qda_y, rf_y, kmeans_y=predict_kmeans(test_x, k))

# Get the votes
prob<-rowSums(y_hat==1)

# Assign label based on max votes
yhat<-ifelse(prob>ncol(yhat)/2,"B","M")

# Accuracy of final model
mean(yhat==test_y)

# Confusion Matrix shows that only two labels were missclassified out of the 115 labels
table(yhat, test_y)

results<-rbind(results, data.frame(method="Ensemble Model(votes)", accuracy=mean(yhat==test_y)))
results%>%knitr::kable()


