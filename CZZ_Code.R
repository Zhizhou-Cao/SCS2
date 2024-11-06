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
humanfeatures <- humanM$features
GPTfeatures <- GPTM$features

#this is a matrix of both human and GPT essays
features <- rbind(humanfeatures, GPTfeatures) 

#here i use author=0 for human, and author=1 for ChatGPT
authornames <- c(rep(0,nrow(humanfeatures)), rep(1,nrow(GPTfeatures)))

# Q1 Human Or Chatgpt?------






