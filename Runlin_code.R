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
  RF_preds[i] <- randomForestCorpus(train_list, test)
}
# Calculate accuracy
DA_accuracy <- sum(DA_preds == authornames) / length(authornames)
DA_accuracy
KNN_accuracy <- sum(KNN_preds==truth_random)/length(truth_random)
KNN_accuracy
RF_accuracy <- sum(RF_preds==truth_random)/length(truth_random)
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



