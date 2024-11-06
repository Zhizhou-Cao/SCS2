library(ggplot2)
library(lattice)
library(class)
library(caret)
library(knitr)

source("stylometryfunctions.R")
source("reducewords.R")

humanM <- loadCorpus("functionwords/functionwords/GPTfunctionwords/","functionwords")
GPTM <- loadCorpus("functionwords/functionwords/humanfunctionwords/","functionwords")

humanfeatures <- humanM$features
GPTfeatures <- GPTM$features


# Create a feature matrix with both human and AI data, where each row represents an essay, and columns represent function word counts
# 0 = human, 1 = Chatgpt
#authornames <- c(rep(0,nrow(humanfeatures)), rep(1,nrow(GPTfeatures)))
authornames <- c(rep(0, length(humanfeatures)), rep(1, length(GPTfeatures)))

# Normalize each feature to account for differences in text length
# for (i in 1:nrow(combine_features)) {
#   combine_features[i, ] <- combine_features[i, ] / sum(combine_features[i, ])
# }

human_matrix <- do.call(rbind, humanfeatures)
GPT_matrix <- do.call(rbind, GPTfeatures)
features <- rbind(human_matrix, GPT_matrix)


for (i in 1:nrow(features)) {
  features[i, ] <- features[i, ] / sum(features[i, ])
}


authornames <- c(rep(0, nrow(human_matrix)), rep(1, nrow(GPT_matrix)))

# Normalize each row to account for text length
features <- t(apply(features, 1, function(row) row / sum(row)))

# Initialize DA_preds
DA_preds <- numeric(nrow(features))
KNN_preds <- numeric(nrow(features))
RF_preds <- numeric(nrow(features))

# Loop through each row for Leave-One-Out Cross-Validation (LOOCV)
for (i in 1:nrow(features)) {
  # Exclude the ith row for training and use it as the test set
  train <- features[-i, ]
  test <- matrix(features[i, ], nrow = 1)
  
  # Convert train matrix into a list of matrices, with each row as a separate matrix
  train_list <- lapply(1:nrow(train), function(j) matrix(train[j, ], nrow = 1))
  
  # Apply discriminantCorpus function
  DA_preds[i] <- discriminantCorpus(train_list, test)
  KNN_preds[i] <- KNNCorpus(train_list, test)
  #RF_preds[i] <- randomForestCorpus(train_list, test)
}
# Calculate accuracy
DA_accuracy <- sum(DA_preds == authornames) / length(authornames)
DA_accuracy
KNN_accuracy <- sum(KNN_preds==authornames)/length(authornames)
KNN_accuracy
RF_accuracy <- sum(RF_preds==authornames)/length(authornames)
RF_accuracy

# DA_preds <- numeric(nrow(features))  # Initialize DA_preds
# for (i in 1:nrow(features)) {
#   # Exclude the ith row for training, and use it as a separate test matrix
#   train <- features[-i, ]
#   test <- matrix(features[i, ], nrow = 1)
#   
#   # Convert train matrix into a list of matrices (each row as a matrix)
#   train_list <- split(train, row(train))
#   train_list <- lapply(train_list, function(row) matrix(row, nrow = 1))
#   
#   # Apply discriminantCorpus function
#   DA_preds[i] <- discriminantCorpus(train_list, test)
# }
# DA_accuracy <- sum(DA_preds == authornames) / length(authornames)


## Random select model

# Random select cross validation (from Zhizhou)
train_random <- features
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
# Multinomial (more than two categories) discriminant analysis
DA_random_accuracy <- sum(predsDA_random==truth_random)/length(truth_random)
DA_random_accuracy
# KNN 
KNN_random_accuracy <- sum(predsKNN_random==truth_random)/length(truth_random)
KNN_random_accurac

