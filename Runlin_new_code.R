library(ggplot2)
library(lattice)
library(class)
library(caret)
library(knitr)

source("stylometryfunctions.R")
source("reducewords.R")

humanM <- loadCorpus("functionwords/functionwords/GPTfunctionwords/","functionwords")
GPTM <- loadCorpus("functionwords/functionwords/humanfunctionwords/","functionwords")

### Create a new dataset for Q1, where we create a new folder containing all the aricles from Ai and all the articles from human respectively
# Specify the path for the new folder
new_folder <- "C:/Users/12145/OneDrive/Desktop/year_4/first_semester/SCS/SCS2/combined_articles/Humans"  # Replace with your desired folder path

# Create the folder
dir.create(new_folder)
# Define the main directory containing the topic folders
main_directory <- "functionwords/functionwords/GPTfunctionwords/"  # Replace with your actual path to "AI" folder
human_main_directory <- "functionwords/functionwords/humanfunctionwords/" 
# Define the destination folder where you want all articles to be copied
gpt_destination_folder <- "combined_articles/GPTs"  # Replace with your actual destination path
human_destination_folder <- "combined_articles/humans"
# Get a list of all article files across all topics
all_files <- list.files(human_main_directory, recursive = TRUE, full.names = TRUE)

# # Ensure the destination folder exists
# if (!dir.exists(human_destination_folder)) {
#   dir.create(human_destination_folder)
# }

# Copy each article to the destination folder
sapply(all_files, function(file) {
  # Get the basename (file name without path) to avoid folder structure in destination
  file_name <- basename(file)
  # Copy the file to the destination folder
  file.copy(file, file.path(human_destination_folder, file_name), overwrite = TRUE)
})

  


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
  predsRF_random <- randomForestCorpus(train_random, test_random)
}
# Multinomial (more than two categories) discriminant analysis
DA_random_accuracy <- sum(predsDA_random==truth_random)/length(truth_random)
DA_random_accuracy
# KNN 
KNN_random_accuracy <- sum(predsKNN_random==truth_random)/length(truth_random)
KNN_random_accuracy
RF_random_accuracy <- sum(predsRF_random==truth_random)/length(truth_random)
RF_random_accuracy




# leave-one cross validation
train_LOOCV <- combined_q1$features
features <- combined_q1$features
test_LOOCV <- NULL
truth_LOOCV <- NULL
predsDA_LOOCV <- NULL
predsKNN_LOOCV <- NULL
for (i in 1:length(train_LOOCV)) {
  for (j in 1:nrow(train_LOOCV[[i]])) {
    test_LOOCV <- matrix(features[[i]][j,],nrow=1)
    train_LOOCV <- features
    train_LOOCV[[i]] <- train_LOOCV[[i]][-j,,drop=FALSE]
    
    pred <- discriminantCorpus(train_LOOCV, test_LOOCV)
    predsDA_LOOCV <- c(predsDA_LOOCV, pred)
    
    pred <- KNNCorpus(train_LOOCV, test_LOOCV)
    predsKNN_LOOCV <- c(predsKNN_LOOCV, pred)
    truth_LOOCV <- c(truth_LOOCV, i)
  }}

# Multinomial (more than two categories) discriminant analysis
DA_LOOCV_accuracy <- sum(predsDA_LOOCV==truth_LOOCV)/length(truth_LOOCV)
DA_LOOCV_accuracy
# KNN
KNN_LOOCV_accuracy <-sum(predsKNN_LOOCV==truth_LOOCV)/length(truth_LOOCV)
KNN_LOOCV_accuracy

# Present the accuracy in table
accuracy_table <- data.frame(
  "One random CV" = c(DA_random_accuracy, KNN_random_accuracy),
  "LOOCV" = c(DA_LOOCV_accuracy, KNN_LOOCV_accuracy),
  row.names = c("Discriminant Analysis", "K-Nearest Neighbors"))
kable(accuracy_table)


# Q2. 测试不同模型在不同数据集的表现
# 三个训练模型【500/750/1000】
# 三个测试集【500/750/1000】
# 出一个3x3的矩阵table

humanM <- loadCorpus("functionwords/functionwords/humanfunctionwords/","functionwords")
GPTM <- loadCorpus("functionwords/functionwords/GPTfunctionwords/","functionwords")
humanM$authornames <- rep(0, length(humanM$authornames))
GPTM$authornames <- rep(1, length(GPTM$authornames))

humanfeatures <- humanM$features
GPTfeatures <- GPTM$features
  
#this is a matrix of both human and GPT essays
combined_q1 <- list(
  features = list(
    human = humanM$features,
    GPTM = GPTM$features),
  authornames = c("human", "GPT"))

numwords_500 <- 500 #number of words to trim the test set down into 500
numwords_750 <- 750
numwords_1000 <- 1000

combined_q1$features$GPTM <- do.call(rbind, combined_q1$features$GPTM)
combined_q1$features$human <- do.call(rbind, combined_q1$features$human)

reducedhumanfeatures_500 <- reducewords(combined_q1$features$human,numwords_500)
reducedGPTfeatures_500 <- reducewords(combined_q1$features$GPTM,numwords_500)

reducedhumanfeatures_750 <- reducewords(combined_q1$features$human,numwords_750)
reducedGPTfeatures_750 <- reducewords(combined_q1$features$GPTM,numwords_750)

reducedhumanfeatures_1000 <- reducewords(combined_q1$features$human,numwords_1000)
reducedGPTfeatures_1000 <- reducewords(combined_q1$features$GPTM,numwords_1000)

combined_500 <- list(
  features = list(
    human = reducedhumanfeatures_500,
    GPTM = reducedGPTfeatures_500),
  authornames = c("human", "GPT"))

combined_750 <- list(
  features = list(
    human = reducedhumanfeatures_750,
    GPTM = reducedGPTfeatures_750),
  authornames = c("human", "GPT"))

combined_1000 <- list(
  features = list(
    human = reducedhumanfeatures_1000,
    GPTM = reducedGPTfeatures_1000),
  authornames = c("human", "GPT"))



# Random select cross validation
train_random_500 <- combined_500$features
train_random_750 <- combined_750$features
train_random_1000 <- combined_1000$features
test_random_500 <- combined_500$features
test_random_750 <- combined_750$features
test_random_1000 <- combined_1000$features
test_random <- NULL
truth_random <- NULL

train_random <- train_random_500
for (i in 1:length(train_random)){
  # select ONE RANDOM book by this author by choosing a row (= book)
  set.seed(1) # Ensure reproducibility of random results
  #testind <- sample(1:nrow(train_random[[i]]), 1)
  testind <- sample(1:nrow(test_random_500[[i]]), 1)
  # add this book to the test set
  test_random <- rbind(test_random, train_random[[i]][testind,])
  truth_random <- c(truth_random, i)
  
  # now discard the book from the training set
  # drop = FALSE prevent the matrix converting into a vector
  train_random[[i]] <- train_random[[i]][-testind,,drop=FALSE]
  
  #predsDA_random <- discriminantCorpus(train_random, test_random)
  predsKNN_random <- KNNCorpus(train_random, test_random)
  #predsRF_random <- randomForestCorpus(train_random, test_random) 
}
# KNN
KNN_random_accuracy <- sum(predsKNN_random==truth_random)/length(truth_random)
KNN_random_accuracy
# Multinomial (more than two categories) discriminant analysis
DA_random_accuracy <- sum(predsDA_random==truth_random)/length(truth_random)
DA_random_accuracy

# Random Forest
predsRF_random_accuracy <- sum(predsRF_random==truth_random)/length(truth_random)
predsRF_random_accuracy


randomeselect_results <- matrix(rep(3, times=9), ncol=3, byrow=TRUE)
#define column names and row names of matrix
colnames(randomeselect_results) <- c('train_500', 'train_750', 'train_1000')
rownames(randomeselect_results) <- c('test_500', 'test_750','test_1000')

#convert matrix to table 
randomeselect_results_KNN <- as.table(randomeselect_results)
randomeselect_results_KNN[1, 1] <- 0.4722222  #[row,column]
#view table 
randomeselect_results_KNN

