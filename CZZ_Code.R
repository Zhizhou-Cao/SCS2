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


