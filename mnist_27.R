########################################################################################
#                                   INTRODUCTION
########################################################################################

# This file has the R code to predict digit recognition based on the MNIST dataset 
# MNIST (Modified National Institute of Standards and Technology) database is 
# a database of handwritten digits created by re-mixing samples from NIST databses
# MNIST consists of 60,000 training images and 10,000 test images

# Postal services use digit recognition to sort mail and need the system to be very accurate 
# to avoid missing mail. 

# The dslabs library has a subset of the MNIST dataset which include images for the
# digits 2 and 7. Each image has 784 pixels that are the predictors for the algorithms.
# The MNIST is a very popular learning exercise for Machine Learning

# The analysis on this dataset was done as part of the course work in 
# HarvardX PH125.9x Certificate Program

# This file uses this subset of 800 images for training and 200 images for testing with
# two predictors x_1, x_2 that the upper_left quadrant pixels of each image.
# The digit recognition system in this analysis uses Supervised Learning Classification
# Algorithms. 

# The information from the images is stored in a matrix format with x_1, x_2 as 
# the two predictors and y the outcome of the prediction

# The Ensemble model with voting system has been implemented here.


########################################################################################


#Loading libraries
library(caret)

# dslabs has a subset of the original MNIST dataset
library(dslabs)


set.seed(1, sample.kind = "Rounding") 

# load dataset
data("mnist_27")

#create a vector to run models
models <- c("glm", "lda", "naive_bayes", "svmLinear", "knn", 
            "gamLoess", "multinom", "qda", "rf", "adaboost")

# Data Exploration
mnist_27$train %>% ggplot(aes(x_1, x_2, color = y)) + geom_point()



#use a function to run models
fits <- lapply(models, function(model){ 
  print(model)
  train(y ~ ., method = model, data = mnist_27$train)
}) 

#name the models
names(fits) <- models

# run predictions on test set
yhat<-sapply(fits, function(fits) {
  predict(fits, mnist_27$test)
})

# Accuracy for each model
data.frame(accuracy=colMeans(yhat==mnist_27$test$y))

# Compute average of accuracy obtained for each model
mean(colMeans(yhat==mnist_27$test$y))

# Compute votes
votes<-rowSums(yhat==7)

# assign max votes label as final label 
new_yhat<-ifelse(votes>length(models)/2, 7,2)

# Accuracy with final label
mean(new_yhat==mnist_27$test$y)


########################################################################################
#                                   CONCLUSION
########################################################################################

# The best model with highest acuracy is the GamLoess method.
# Gam Loess is linear regression but with a complex function instead of linear

# The ensemble model decreased from 84.5& of GamLoess to 81.5% but can be considered
# a much more robust model for this given data.


