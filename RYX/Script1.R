# Import Libraries
library(ggplot2)
library(lattice)
library(class)
library(caret)
library(knitr)



# Load the functions
source("stylometryfunctions.R")
source("reducewords.R")

# numwords <- 500 #number of words to trim the test set down into
# topic <- 6 #Architecture
humanM <- loadCorpus("functionwords/functionwords/humanfunctionwords/","functionwords")
GPTM <- loadCorpus("functionwords/functionwords/GPTfunctionwords/","functionwords")

#humanfeatures <- humanM$features #select the essays on this particular topic
#GPTfeatures <- GPTM$features

humanfeatures <- do.call(rbind, humanM$features)
GPTfeatures <-  do.call(rbind, GPTM$features)

features <- rbind(humanfeatures, GPTfeatures) #this is a matrix of both human and GPT essays on Topic

#here i use author=0 for human, and author=1 for ChatGPT
authornames <- c(rep(1,nrow(humanfeatures)), rep(2,nrow(GPTfeatures)))

#now reduce the essays down to numwords words
# reducedhumanfeatures <- reducewords(humanfeatures,numwords)
# reducedGPTfeatures <- reducewords(GPTfeatures,numwords)
# reducedfeatures <- rbind(reducedhumanfeatures, reducedGPTfeatures)

DA_pred <- numeric(nrow(features))
KNN_pred <- numeric(nrow(features))

# LOOCV
for (i in 1:nrow(features)) {
  train <- features[-i,]
  test <- matrix(features[i,,drop=FALSE], nrow = 1) #note that only the test set size changes

  KNN_pred[i] <- myKNN(train, test, authornames[-i], k = 1)
  
  # Prepare data format for discriminantCorpus
  train_list <- split(train, f = authornames[-i])  # Split training data into two classes based on author labels
  DA_pred[i] <- discriminantCorpus(train_list, test)
  
}

print(KNN_pred)

KNN_LOOCV_accuracy <-sum(KNN_pred==authornames)/length(authornames)
KNN_LOOCV_accuracy


