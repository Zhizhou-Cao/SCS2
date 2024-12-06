# Import Libraries
library(ggplot2)
library(lattice)
library(class)
library(caret)
library(knitr)

source("stylometryfunctions.R")
source("reducewords.R")

humanM <- loadCorpus("functionwords/functionwords/humanfunctionwords/","functionwords")
GPTM <- loadCorpus("functionwords/functionwords/GPTfunctionwords/","functionwords")

humanM$authornames <- rep(0, length(humanM$authornames))
GPTM$authornames <- rep(1, length(GPTM$authornames))



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
}
# Multinomial (more than two categories) discriminant analysis
DA_random_accuracy <- sum(predsDA_random==truth_random)/length(truth_random)
DA_random_accuracy
# KNN 
KNN_random_accuracy <- sum(predsKNN_random==truth_random)/length(truth_random)
KNN_random_accuracy





# # leave-one cross validation
# train_LOOCV <- combined_q1$features
# features <- combined_q1$features
# test_LOOCV <- NULL
# truth_LOOCV <- NULL
# predsDA_LOOCV <- NULL
# predsKNN_LOOCV <- NULL
# predsRF_LOOCV <- NULL
# for (i in 1:length(train_LOOCV)) {
#   for (j in 1:nrow(train_LOOCV[[i]])) {
#     test_LOOCV <- matrix(features[[i]][j,],nrow=1)
#     train_LOOCV <- features
#     train_LOOCV[[i]] <- train_LOOCV[[i]][-j,,drop=FALSE]
#     
#     pred <- discriminantCorpus(train_LOOCV, test_LOOCV)
#     predsDA_LOOCV <- c(predsDA_LOOCV, pred)
#     
#     pred <- KNNCorpus(train_LOOCV, test_LOOCV)
#     predsKNN_LOOCV <- c(predsKNN_LOOCV, pred)
#     
#     pred <- randomForestCorpus(train_LOOCV, test_LOOCV)
#     predsRF_LOOCV <- c(predsRF_LOOCV, pred)
#     
#     truth_LOOCV <- c(truth_LOOCV, i)
#   }}

# Multinomial (more than two categories) discriminant analysis
DA_LOOCV_accuracy <- sum(predsDA_LOOCV==truth_LOOCV)/length(truth_LOOCV)
DA_LOOCV_accuracy
# KNN
KNN_LOOCV_accuracy <-sum(predsKNN_LOOCV==truth_LOOCV)/length(truth_LOOCV)
KNN_LOOCV_accuracy

RF_LOOCV_accuracy <-sum(predsRF_LOOCV==truth_LOOCV)/length(truth_LOOCV)
RF_LOOCV_accuracy

# # Present the accuracy in table
# accuracy_table <- data.frame(
#   "One random CV" = c(DA_random_accuracy, KNN_random_accuracy),
#   "LOOCV" = c(DA_LOOCV_accuracy, KNN_LOOCV_accuracy),
#   row.names = c("Discriminant Analysis", "K-Nearest Neighbors"))
# kable(accuracy_table)




