# The tissue_gene_expression is a dataset that is a subset of 
# the genomicsclass GitHub repository. This dataset contains
# 500 genes that were selected randomly.
# Each row is a gene expression profile and each column is
# a different gene. The column names are the gene symbols.
# The outcome y is a character vector representing one 
# of the seven tissue types. The goal of the analysis is to 
# predict the tissue type

# The analysis on this dataset was done as part of the course work in 
# HarvardX PH125.9x Certificate Program


data("tissue_gene_expression")
dim(tissue_gene_expression$x)

y<-tissue_gene_expression$y
x<-as.matrix(tissue_gene_expression$x)


########################################################
# Principal Component Analysis

pca <- prcomp(x)

# Summary of the PCA analysis shows that the 500 features can 
# be reduced to 189 features of which the first 25 of them 
# contribute to 90% of the variation
summary(pca)

# PC1 Vs PC2 plot
data.frame(PC1=pca$x[,1], PC2=pca$x[,2], tissue=tissue_gene_expression$y) %>% 
  ggplot(aes(PC1,PC2, color=tissue))+
  geom_point()


# The first 10 PC Box plot show the variation in the distribution
boxplot(pca$x[,1:10], color=tissue_gene_expression$y, label=TRUE)
boxplot(pca$x[,7])

# Individual boxplots of the first 10 PCs show the variation in 
# each tissue type
for(i in 1:10){
  boxplot(pc$x[,i] ~ tissue_gene_expression$y, main = paste("PC", i))
}


###############################################

# CLustering Segmentation

d <- dist(tissue_gene_expression$x)
h <- hclust(d)

# Segmentation plot
plot(h, cex = 0.65, main = "", xlab = "")

# Kmeans clustering for 7 tissue types
kmeans_out<-kmeans(tissue_gene_expression$x, centers=7)

# Clustering Groups
groups<-kmeans_out$cluster

# The classification clustering table below shows that
# tissue types cerebellum, hippocampus, kidney and liver
# have been clustered into multiple groups
table(kmeans_out$cluster, tissue_gene_expression$y)

###################################################################
# Create Train Test sets for algorithms

train_index <- createDataPartition(tissue_gene_expression$y, times = 1, p = 0.8, list = FALSE)
test_set_x <- as.data.frame(x[train_index,])
train_set_x <- as.data.frame(x[-train_index,])
test_set_y <- y[train_index]
train_set_y <- y[-train_index]

#########################################################
# KNN Model

# find best k value for KNN model, maximize accuracy
ks<- c(1,3,5,7,9,11)
result <- sapply(ks, function(ks) {
  knn_fit<-knn3(train_set_x,train_set_y, k=ks)
  y_hat<-predict(knn_fit, test_set_x, type="class") 
  confusionMatrix(data=y_hat, reference=test_set_y)$overall["Accuracy"]
})

data.frame(k=ks, Accuracy=result)

# summarize results in a data frame to compare
results<-data.frame(method="KNN", Accuracy=max(result))


#########################################################
# LDA Model

# train function from caret package for LDA model
train_lda<-train(train_set_x,train_set_y, method="lda")

# predict outcomes
lda_y<-predict(train_lda, test_set_x)

# calculate accuracy
lda_acc<-mean(lda_y==test_set_y)

# confusion matrix
table(lda_y, test_set_y)

# Add results
results<-rbind(results, data.frame(method="LDA", Accuracy=lda_acc))

#########################################################
# CART Model

# train function from caret package for CART model, fine tune cp
tree_model <- train(train_set_x, train_set_y, method = "rpart",
                   tuneGrid = data.frame(cp = seq(0, 0.5, 0.05)))

# cp Vs accuracy
ggplot(tree_model)

# confusion matrix
confusionMatrix(tree_model)

#Tree Model
plot(tree_model$finalModel, margin = 0.1)
text(tree_model$finalModel, cex = 0.75)

# predictions & accuracy calculation
tree_y<-predict(tree_model, test_set_x)
tree_acc<-mean(tree_y==test_set_y)
table(tree_y, test_set_y)

# add results
results<-rbind(results, data.frame(method="CART", Accuracy=tree_acc))

#########################################################
# Random Forest Model

# no. of variables at each node
mt<-seq(50, 200, 25)

# train function from caret package for CART model, fine tune mtry
rf_model <- train(train_set_x, train_set_y, method = "rf", nodesize = 1,
                    tuneGrid = data.frame(mtry=mt))

# predictions & accuracy calculation
rf_y<-predict(rf_model, test_set_x)
rf_acc<-mean(rf_y==test_set_y)
table(rf_y, test_set_y)

ggplot(rf_model)

# get variable importance
imp<-varImp(rf_model)
data_frame(term = rownames(imp$importance), 
           importance = imp$importance$Overall) %>%
  mutate(rank = rank(-importance)) %>% arrange(desc(importance)) 

# add results
results<-rbind(results, data.frame(method="Random forest", Accuracy=rf_acc))

# Final results show that KNN model showed best results with k=1, however, 
# k=1 means that each observation is by itself and this is usually an example of overtraining
# the next best model is the Radom Forest model with accuracy of 89.6%
