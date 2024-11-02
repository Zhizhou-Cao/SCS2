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


DA_preds <- numeric(nrow(features))  # Initialize DA_preds
for (i in 1:nrow(features)) {
  # Exclude the ith row for training, and use it as a separate test matrix
  train <- features[-i, ]
  test <- matrix(features[i, ], nrow = 1)
  
  # Convert train matrix into a list of matrices (each row as a matrix)
  train_list <- split(train, row(train))
  train_list <- lapply(train_list, function(row) matrix(row, nrow = 1))
  
  # Apply discriminantCorpus function
  DA_preds[i] <- discriminantCorpus(train_list, test)
}
#DA_accuracy <- sum(DA_preds == authornames) / length(authornames)



