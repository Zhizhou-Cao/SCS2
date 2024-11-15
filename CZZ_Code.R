# Import Libraries
library(ggplot2)
library(lattice)
library(class)
library(caret)
library(knitr)
library(randomForest)
source("stylometryfunctions.R")
source("reducewords.R")

humanM <- loadCorpus("functionwords/functionwords/humanfunctionwords/","functionwords")
GPTM <- loadCorpus("functionwords/functionwords/GPTfunctionwords/","functionwords")

humanM$authornames <- rep(0, length(humanM$authornames))
GPTM$authornames <- rep(1, length(GPTM$authornames))


# Q1 ----
# Create a new list for Q1
combined_q1 <- list(
  features = list(
    human = humanM$features,
    GPTM = GPTM$features),
  authornames = c("human", "GPT"))


combined_q1$features$GPTM <- do.call(rbind, combined_q1$features$GPTM)
combined_q1$features$human <- do.call(rbind, combined_q1$features$human)


# Random select cross validation
train_random <- combined_q1$features
test_random <- NULL
truth_random <- NULL

for (i in 1:length(train_random)){
  # select ONE RANDOM book by this author by choosing a row (= book)
  set.seed(1) # Ensure reproducibility of random results
  testind <- sample(1:nrow(train_random[[i]]), 1)
  # add this book to the test set
  test_random <- rbind(test_random, train_random[[i]][testind,])
  truth_random <- c(truth_random, i)
  
  # now discard the book from the training set
  # drop = FALSE prevent the matrix converting into a vector
  train_random[[i]] <- train_random[[i]][-testind,,drop=FALSE]
  
  predsDA_random <- discriminantCorpus(train_random, test_random)
  predsKNN_random <- KNNCorpus(train_random, test_random)
  predsRF_random <- randomForestCorpus(train_random, test_random) 
}
# Multinomial (more than two categories) discriminant analysis
DA_random_accuracy <- sum(predsDA_random==truth_random)/length(truth_random)
DA_random_accuracy
# KNN 
KNN_random_accuracy <- sum(predsKNN_random==truth_random)/length(truth_random)
KNN_random_accuracy
# Random Forest
predsRF_random_accuracy <- sum(predsRF_random==truth_random)/length(truth_random)
predsRF_random_accuracy




# leave-one cross validation
train_LOOCV <- combined_q1$features
features <- combined_q1$features

test_LOOCV <- NULL
truth_LOOCV <- NULL
predsDA_LOOCV <- NULL
predsKNN_LOOCV <- NULL
predsRF_LOOCV <- NULL

for (i in 1:length(train_LOOCV)) {
  for (j in 1:nrow(train_LOOCV[[i]])) {
    test_LOOCV <- matrix(features[[i]][j,],nrow=1)
    train_LOOCV <- features
    train_LOOCV[[i]] <- train_LOOCV[[i]][-j,,drop=FALSE]
    
    pred <- discriminantCorpus(train_LOOCV, test_LOOCV)
    predsDA_LOOCV <- c(predsDA_LOOCV, pred)
    
    pred <- KNNCorpus(train_LOOCV, test_LOOCV)
    predsKNN_LOOCV <- c(predsKNN_LOOCV, pred)
    
    pred <- randomForestCorpus(train_LOOCV, test_LOOCV)
    predsRF_LOOCV <- c(predsRF_LOOCV, pred)
    
    truth_LOOCV <- c(truth_LOOCV, i)
  }}

# Multinomial (more than two categories) discriminant analysis
DA_LOOCV_accuracy <- sum(predsDA_LOOCV==truth_LOOCV)/length(truth_LOOCV)
DA_LOOCV_accuracy
# KNN
KNN_LOOCV_accuracy <-sum(predsKNN_LOOCV==truth_LOOCV)/length(truth_LOOCV)
KNN_LOOCV_accuracy
# RF
RF_LOOCV_accuracy <-sum(predsRF_LOOCV==truth_LOOCV)/length(truth_LOOCV)
RF_LOOCV_accuracy



# Present the accuracy in table
accuracy_table <- data.frame(
  "One random CV" = c(DA_random_accuracy, KNN_random_accuracy),
  "LOOCV" = c(DA_LOOCV_accuracy, KNN_LOOCV_accuracy),
  row.names = c("Discriminant Analysis", "K-Nearest Neighbors"))
kable(accuracy_table)




# K-Fold (没做出来)

k <-3 # k value for k-nearest neighbours
numfolds <- 5 # Number of fold for validation
folds <- createFolds(combined_q1$features, k=numfolds)
accuracies <- numeric(numfolds)

# Initialize empty vectors for predictions and true labels
predsDA_KFold <- NULL
predsKNN_KFold <- NULL
predsRF_KFold <- NULL
truth_KFold <- NULL

for (i in 1:numfolds) {
  # Define training and test sets
  test_indices <- folds[[i]]
  train_indices <- setdiff(seq_along(combined_q1$features), test_indices)
  
  # Extract features for training and testing
  train_features <- combined_q1$features[train_indices]
  test_features <- combined_q1$features[test_indices]
  test_truth <- rep(seq_along(test_indices), length(test_indices))
  
  # Predictions for each model
  predDA <- discriminantCorpus(train_features, test_features)
  predKNN <- KNNCorpus(train_features, test_features)
  predRF <- randomForestCorpus(train_features, test_features)
  
  # Store predictions and true labels
  predsDA_KFold <- c(predsDA_KFold, predDA)
  predsKNN_KFold <- c(predsKNN_KFold, predKNN)
  predsRF_KFold <- c(predsRF_KFold, predRF)
  truth_KFold <- c(truth_KFold, test_truth)
}

# Calculate accuracy for each model
DA_KFold_accuracy <- sum(predsDA_KFold == truth_KFold) / length(truth_KFold)
KNN_KFold_accuracy <- sum(predsKNN_KFold == truth_KFold) / length(truth_KFold)
RF_KFold_accuracy <- sum(predsRF_KFold == truth_KFold) / length(truth_KFold)


 



# Define the number of folds
k <- 5  # You can adjust this based on the dataset size and computational power
folds <- createFolds(combined_q1$features, k = k)
accuracies <- numeric(k)
# Initialize empty vectors for predictions and true labels
predsDA_KFold <- NULL
predsKNN_KFold <- NULL
predsRF_KFold <- NULL
truth_KFold <- NULL

# Loop through each fold
for (i in 1:k) {
  traindata <- combined_q1[-folds[[i]],-5]
  trainlabels <- combined_q1[-folds[[i]],5]
  testdata <- combined_q1[folds[[i]],-5]
  testlabels <- combined_q1[folds[[i]],5]
  predKNN <- KNNCorpus(traindata,testdata)
  accuracies[i] <- mean(preds==testlabels)
}

# Calculate accuracy for each model
DA_KFold_accuracy <- sum(predsDA_KFold == truth_KFold) / length(truth_KFold)
KNN_KFold_accuracy <- sum(predsKNN_KFold == truth_KFold) / length(truth_KFold)
RF_KFold_accuracy <- sum(predsRF_KFold == truth_KFold) / length(truth_KFold)
















#不可用----

# 合并列表
combined_features <- c(humanM$features, GPTM$features)
combined_authornames <- c(humanM$authornames, GPTM$authornames)
combined_booknames <- c(humanM$booknames, GPTM$booknames)
combined_list <- list(features = combined_features, 
                      authornames = combined_authornames, 
                      booknames = combined_booknames)

#select all essays
#humanfeatures <- humanM$features[[1]]
#GPTfeatures <- GPTM$features[[1]]

#this is a matrix of both human and GPT essays
#features <- rbind(humanfeatures, GPTfeatures) 

#here i use author=0 for human, and author=1 for ChatGPT
#authornames <- c(rep(0,length(humanfeatures)), rep(1,length(GPTfeatures)))
# Q1 Human Or Chatgpt?------



human_matrix <- do.call(rbind, humanfeatures)
GPT_matrix <- do.call(rbind, GPTfeatures)
features <- rbind(human_matrix, GPT_matrix)
authornames <- c(rep(0,nrow(human_matrix)), rep(1,nrow(GPT_matrix)))

for (i in 1:nrow(features)) {
  features[i, ] <- features[i, ] / sum(features[i, ])
}

HG <- list(features = list(features), authornames = list(authornames))



# Random select cross validation
train_random <- combined_list$features
test_random <- NULL
truth_random <- NULL

for (i in 1:length(train_random)){
  # select ONE RANDOM book by this author by choosing a row (= book)
  set.seed(1) # Ensure reproducibility of random results
  testind <- sample(1:nrow(train_random[[i]]), 1)
  # add this book to the test set
  test_random <- rbind(test_random, train_random[[i]][testind,])
  truth_random <- c(truth_random, i)
  
  # now discard the book from the training set
  # drop = FALSE prevent the matrix converting into a vector
  train_random[[i]] <- train_random[[i]][-testind,,drop=FALSE]
  
  predsDA_random <- discriminantCorpus(train_random, test_random)
  predsKNN_random <- KNNCorpus(train_random, test_random)
}

truth_random1 <- unlist(lapply(truth_random, function(x) ifelse(x >= 0 & x <= 110, 0, 1)))
predsDA_random1 <- unlist(lapply(predsDA_random, function(x) ifelse(x >= 0 & x <= 110, 0, 1)))
predsKNN_random1 <- unlist(lapply(predsKNN_random, function(x) ifelse(x >= 0 & x <= 110, 0, 1)))
# Multinomial (more than two categories) discriminant analysis
DA_random_accuracy <- sum(predsDA_random1==truth_random1)/length(truth_random1)
DA_random_accuracy
# KNN 
KNN_random_accuracy <- sum(predsKNN_random1==truth_random1)/length(truth_random1)
KNN_random_accuracy

