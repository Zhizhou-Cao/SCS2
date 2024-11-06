# 谨慎修改代码！！！
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

# 合并列表
combined_features <- c(humanM$features, GPTM$features)
combined_authornames <- c(humanM$authornames, GPTM$authornames)
combined_booknames <- c(humanM$booknames, GPTM$booknames)
combined_list <- list(features = combined_features, 
                      authornames = combined_authornames, 
                      booknames = combined_booknames)

# Create a new list for Q1
combined_q1 <- list(
  features = list(
    human = humanM$features,
    GPTM = GPTM$features
  ),
  authornames = c(0, 1)
)