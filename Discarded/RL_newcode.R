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


# original
combined_q1 <- list(
  features = list(
    human = humanM$features,
    GPTM = GPTM$features),
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





# K-Fold (by CZZ)
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

# Define the function
evaluate_models <- function(trainset, testset, n_folds = 5) {
  # Set up cross-validation parameters
  set.seed(1)  # Ensure reproducibility
  
  # Initialize accuracy lists
  DAaccuracy_list <- numeric(n_folds)
  KNNaccuracy_list <- numeric(n_folds)
  RFaccuracy_list <- numeric(n_folds)
  
  # Create folds for cross-validation
  folds_human <- sample(rep(1:n_folds, length.out = nrow(trainset$human)))
  folds_gptm <- sample(rep(1:n_folds, length.out = nrow(trainset$GPTM)))
  
  # Perform cross-validation
  for (fold in 1:n_folds) {
    # Training and testing split for human samples
    train_human <- trainset$human[folds_human != fold, , drop = FALSE]
    test_human <- testset$human[folds_human == fold, , drop = FALSE]
    
    # Training and testing split for GPTM samples
    train_gptm <- trainset$GPTM[folds_gptm != fold, , drop = FALSE]
    test_gptm <- testset$GPTM[folds_gptm == fold, , drop = FALSE]
    
    # Combine training and test sets
    train_fold <- list(train_human, train_gptm)
    test_fold <- rbind(test_human, test_gptm)
    
    # Create ground truth for the test set
    truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))
    
    # Train and predict using the models
    predsDA_fold <- discriminantCorpus(train_fold, test_fold)
    predsKNN_fold <- KNNCorpus(train_fold, test_fold)
    predsRF_fold <- randomForestCorpus(train_fold, test_fold)
    
    # Calculate accuracy for each model
    DAaccuracy_list[fold] <- sum(predsDA_fold == truth_fold) / length(truth_fold)
    KNNaccuracy_list[fold] <- sum(predsKNN_fold == truth_fold) / length(truth_fold)
    RFaccuracy_list[fold] <- sum(predsRF_fold == truth_fold) / length(truth_fold)
  }
  
  # Calculate mean accuracy across all folds
  DA_mean_accuracy <- mean(DAaccuracy_list)
  KNN_mean_accuracy <- mean(KNNaccuracy_list)
  RF_mean_accuracy <- mean(RFaccuracy_list)
  
  # Return the accuracies
  return(list(
    DA_accuracy = DA_mean_accuracy,
    KNN_accuracy = KNN_mean_accuracy,
    RF_accuracy = RF_mean_accuracy
  ))
}
# Example usage of the function
results <- evaluate_models(trainset = combined_1000$features, testset = combined_1000$features)
print(results)

# Define the trainsets and testsets
trainsets <- list(combined_500$features, combined_750$features, combined_1000$features)
testsets <- list(combined_500$features, combined_750$features, combined_1000$features)

# Initialize matrices for each method
DA_matrix <- matrix(0, nrow = 3, ncol = 3, 
                    dimnames = list(c("Train-500", "Train-750", "Train-1000"), 
                                    c("Test-500", "Test-750", "Test-1000")))

KNN_matrix <- matrix(0, nrow = 3, ncol = 3, 
                     dimnames = list(c("Train-500", "Train-750", "Train-1000"), 
                                     c("Test-500", "Test-750", "Test-1000")))

RF_matrix <- matrix(0, nrow = 3, ncol = 3, 
                    dimnames = list(c("Train-500", "Train-750", "Train-1000"), 
                                    c("Test-500", "Test-750", "Test-1000")))

# Loop over trainsets and testsets
for (i in 1:length(trainsets)) {
  for (j in 1:length(testsets)) {
    # Evaluate the models
    results <- evaluate_models(trainsets[[i]], testsets[[j]])
    
    # Store results in respective matrices
    DA_matrix[i, j] <- results$DA_accuracy
    KNN_matrix[i, j] <- results$KNN_accuracy
    RF_matrix[i, j] <- results$RF_accuracy
  }
}

# Print the matrices
print("DA Accuracy Matrix:")
print(DA_matrix)

print("KNN Accuracy Matrix:")
print(KNN_matrix)

print("RF Accuracy Matrix:")
print(RF_matrix)


# Q3 two models(train: 1.whole train set/ 2.except "Architecture").  test set always Architecture

# See which topic has the most papers
# Define the path to the main directory
main_directory <- "functionwords/titles"  

# Get a list of all subdirectories in the main directory
subdirs <- list.dirs(main_directory, recursive = FALSE)

# Count the number of files in each subdirectory
subfile_counts <- sapply(subdirs, function(dir) {
  length(list.files(dir))
})

# Find the directory with the most files
max_files <- max(subfile_counts)
max_dir <- names(subfile_counts)[which(subfile_counts == max_files)]
min_files <- min(subfile_counts)
min_dir <- names(subfile_counts)[which(subfile_counts == min_files)]
# Print the result
cat("The folder with the most subfiles is:", max_dir, "with", max_files, "files.\n")
cat("The folder with the most subfiles is:", min_dir, "with", min_files, "files.\n")
max_index <- which.max(subfile_counts)   
max_index  # 'Stories and literature' has 97 papers
min_index <- which.min(subfile_counts)
min_index


# Q3.1 train: 纯"Architecture"; test: "Architecture"
# Number of papers and sets
num_papers <- 73
num_sets <- 5

# Divide papers into 5 sets (approximately equal)
set.seed(1)
set_indices <- sample(rep(1:num_sets, length.out = num_papers))  # Randomly assign papers to sets

# Choose one of the 5 sets as the test set
test_set_id <- 1  # You can change this to select a different set as the test set
test_indices <- which(set_indices == test_set_id)

# Indices for the remaining sets
remaining_sets <- setdiff(1:num_sets, test_set_id)

# Initialize matrix to store accuracies
methods <- c("DA", "KNN", "RF")
accuracy_matrix <- matrix(0, nrow = length(remaining_sets), ncol = length(methods),
                          dimnames = list(paste0("TrainSets=", 1:length(remaining_sets)), methods))

# Test data (fixed throughout the process)
test_human <- combined_only_story$features$human[test_indices, , drop = FALSE]
test_gptm <- combined_only_story$features$GPTM[test_indices, , drop = FALSE]
test_data <- rbind(test_human, test_gptm)

# Ground truth for the test data
truth_test <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))

# Loop through 1 to 4 trainsets
for (train_size in 1:length(remaining_sets)) {
  # Select train indices by combining combined_without_story with train_size sets from the remaining sets
  train_set_ids <- remaining_sets[1:train_size]
  additional_train_indices <- which(set_indices %in% train_set_ids)
  
  # Train data
  train_human <- combined_only_story$features$human[additional_train_indices, , drop = FALSE]
  train_gptm <-  combined_only_story$features$GPTM[additional_train_indices, , drop = FALSE]
  train_data <- list(human = train_human, GPTM = train_gptm)
  
  # Perform DA, KNN, and RF
  predsDA <- discriminantCorpus(train_data, test_data)
  predsKNN <- KNNCorpus(train_data, test_data)
  predsRF <- randomForestCorpus(train_data, test_data)
  
  # Calculate and store accuracies
  accuracy_matrix[train_size, "DA"] <- sum(predsDA == truth_test) / length(truth_test)
  accuracy_matrix[train_size, "KNN"] <- sum(predsKNN == truth_test) / length(truth_test)
  accuracy_matrix[train_size, "RF"] <- sum(predsRF == truth_test) / length(truth_test)
}

# Print the accuracy matrix
print("Accuracy Matrix by Trainset Sizes:")
print(accuracy_matrix)




# Q3.2 train: 全部剩下 ; test: "Architecture"
topic <- 15 #Stories and literature
humanfeatures_story <- humanM$features[[topic]] #select the essays on this particular topic
GPTfeatures_story <- GPTM$features[[topic]]

features_only_story <- rbind(humanfeatures, GPTfeatures) #this is a matrix of both human and GPT essays
authornames_story <- c(rep(0,nrow(humanfeatures)), rep(1,nrow(GPTfeatures)))

combined_only_story <- list(
  features = list(
    human = humanfeatures_story,
    GPTM = GPTfeatures_story),
  authornames = c("human", "GPT"))


humanfeatures_without_story <- humanM$features[-topic]
GPTfeatures_without_story <- GPTM$features[-topic]

# Combine the updated features into a new dataset
combined_without_story <- list(
  features = list(
    human = do.call(rbind, humanfeatures_without_story),  # Combine into a matrix
    GPTM = do.call(rbind, GPTfeatures_without_story)      # Combine into a matrix
  ),
  authornames = c("human", "GPT")
)

# Train and Test Datasets
train_human <- combined_without_story$features$human
train_gptm <- combined_without_story$features$GPTM
test_human <- combined_only_story$features$human
test_gptm <- combined_only_story$features$GPTM

# Combine Train Data
train_data <- list(
  human = train_human,
  GPTM = train_gptm
)

# Combine Test Data
test_data <- rbind(
  test_human,
  test_gptm
)
# Ground truth for the test data (1 for human, 2 for GPTM)
truth_test <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))

# Perform Discriminant Analysis (DA)
predsDA <- discriminantCorpus(train_data, test_data)

# Perform K-Nearest Neighbors (KNN)
predsKNN <- KNNCorpus(train_data, test_data)

# Perform Random Forest (RF)
predsRF <- randomForestCorpus(train_data, test_data)

# Evaluate Accuracy
DA_accuracy <- sum(predsDA == truth_test) / length(truth_test)
KNN_accuracy <- sum(predsKNN == truth_test) / length(truth_test)
RF_accuracy <- sum(predsRF == truth_test) / length(truth_test)

# Print Results
cat("Discriminant Analysis Accuracy:", DA_accuracy, "\n")
cat("K-Nearest Neighbors Accuracy:", KNN_accuracy, "\n")
cat("Random Forest Accuracy:", RF_accuracy, "\n")

# Q3.3 train: 全部剩下 + "Architecture"; test: "Architecture"

# Number of papers and sets
num_papers <- 73
num_sets <- 5

# Divide papers into 5 sets (approximately equal)
set.seed(1)
set_indices <- sample(rep(1:num_sets, length.out = num_papers))  # Randomly assign papers to sets

# Choose one of the 5 sets as the test set
test_set_id <- 1  # You can change this to select a different set as the test set
test_indices <- which(set_indices == test_set_id)

# Indices for the remaining sets
remaining_sets <- setdiff(1:num_sets, test_set_id)

# Initialize matrix to store accuracies
methods <- c("DA", "KNN", "RF")
accuracy_matrix <- matrix(0, nrow = length(remaining_sets), ncol = length(methods),
                          dimnames = list(paste0("TrainSets=", 1:length(remaining_sets)), methods))

# Test data (fixed throughout the process)
test_human <- combined_only_story$features$human[test_indices, , drop = FALSE]
test_gptm <- combined_only_story$features$GPTM[test_indices, , drop = FALSE]
test_data <- rbind(test_human, test_gptm)

# Ground truth for the test data
truth_test <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))

# Loop through 1 to 4 trainsets
for (train_size in 1:length(remaining_sets)) {
  # Select train indices by combining combined_without_story with train_size sets from the remaining sets
  train_set_ids <- remaining_sets[1:train_size]
  additional_train_indices <- which(set_indices %in% train_set_ids)
  
  # Train data
  train_human <- rbind(combined_without_story$features$human,
                       combined_only_story$features$human[additional_train_indices, , drop = FALSE])
  train_gptm <- rbind(combined_without_story$features$GPTM,
                      combined_only_story$features$GPTM[additional_train_indices, , drop = FALSE])
  train_data <- list(human = train_human, GPTM = train_gptm)
  
  # Perform DA, KNN, and RF
  predsDA <- discriminantCorpus(train_data, test_data)
  predsKNN <- KNNCorpus(train_data, test_data)
  predsRF <- randomForestCorpus(train_data, test_data)
  
  # Calculate and store accuracies
  accuracy_matrix[train_size, "DA"] <- sum(predsDA == truth_test) / length(truth_test)
  accuracy_matrix[train_size, "KNN"] <- sum(predsKNN == truth_test) / length(truth_test)
  accuracy_matrix[train_size, "RF"] <- sum(predsRF == truth_test) / length(truth_test)
}

# Print the accuracy matrix
print("Accuracy Matrix by Trainset Sizes:")
print(accuracy_matrix)




# # Q3.1 another way
# 
# topic <- 60 #Stories and literature
# humanfeatures_story <- humanM$features[[topic]] #select the essays on this particular topic
# GPTfeatures_story <- GPTM$features[[topic]]
# 
# features_only_story <- rbind(humanfeatures, GPTfeatures) #this is a matrix of both human and GPT essays
# authornames_story <- c(rep(0,nrow(humanfeatures)), rep(1,nrow(GPTfeatures)))
# 
# combined_only_arch <- list(
#   features = list(
#     human = humanfeatures_story,
#     GPTM = GPTfeatures_story),
#   authornames = c("human", "GPT"))
# 
# 
# train_random <- combined_only_arch$features
# 
# # Set up cross-validation parameters
# set.seed(1) # Ensure reproducibility of folds
# n_folds <- 5
# # Store accuracy for each fold
# DAaccuracy_list <- numeric(n_folds)
# KNNaccuracy_list <- numeric(n_folds)
# RFaccuracy_list <- numeric(n_folds)
# # Create folds for each class (human and GPTM)
# folds_human <- sample(rep(1:n_folds, length.out = nrow(train_random$human)))
# folds_gptm <- sample(rep(1:n_folds, length.out = nrow(train_random$GPTM)))
# 
# # Perform 5-fold cross-validation
# for (fold in 1:n_folds) {
#   # Training and testing data split for human samples
#   train_human <- train_random$human[folds_human != fold, , drop = FALSE]
#   test_human <- train_random$human[folds_human == fold, , drop = FALSE]
# 
#   # Training and testing data split for GPTM samples
#   train_gptm <- train_random$GPTM[folds_gptm != fold, , drop = FALSE]
#   test_gptm <- train_random$GPTM[folds_gptm == fold, , drop = FALSE]
# 
#   # Combine training and test sets
#   train_fold <- list(train_human, train_gptm)
#   test_fold <- rbind(test_human, test_gptm)
# 
#   # Create ground truth for the test set
#   truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))
# 
#   # Train and predict using discriminant analysis
#   predsDA_fold <- discriminantCorpus(train_fold, test_fold)
#   predsKNN_fold <- KNNCorpus(train_fold, test_fold)
#   predsRF_fold <- randomForestCorpus(train_fold, test_fold)
#   # Calculate accuracy for the fold
#   DAaccuracy_list[fold] <- sum(predsDA_fold == truth_fold) / length(truth_fold)
#   KNNaccuracy_list[fold] <- sum(predsKNN_fold == truth_fold) / length(truth_fold)
#   RFaccuracy_list[fold] <- sum(predsRF_fold == truth_fold) / length(truth_fold)
# }
# 
# # Average accuracy across all folds
# DAmean_accuracy <- mean(DAaccuracy_list)
# DAmean_accuracy
# KNNmean_accuracy <- mean(KNNaccuracy_list)
# KNNmean_accuracy
# RFmean_accuracy <- mean(RFaccuracy_list)
# RFmean_accuracy


# 3.4 investigate how the accuracy changes when you only use a smaller number of function
# words

# Calculate variance for each function word

# The Elbow position is 3, so we take the first 3 function words that has the most variance
# For human features
human_all_words <- combined_q1$features$human

# For GPTM features
gptm_all_words <- combined_q1$features$GPTM

# Combine them if needed
combined_all_features <- rbind(human_all_words, gptm_all_words)

variances <- apply(combined_all_features, 2, var)
variances
variances <- variances[-71]
#print(variances)
top_words <- order(variances, decreasing = TRUE)  # Replace N with the desired number
top_words


# Extract the variance values of the top 5 function words
top_variances <- variances[top_words]
# Print the variances
print(top_variances)

variance_df <- data.frame(
  Rank = 1:length(variances),
  Variance = sorted_variances
)

points_to_annotate <- c(3, 6, 15, 70)
# Add annotations to the plot
ggplot(variance_df, aes(x = Rank, y = Variance)) +
  geom_line() +
  geom_point() +
  geom_text(data = variance_df[points_to_annotate, ], 
            aes(label = Rank), 
            vjust = -0.5, 
            color = "red") +  # Adjust vjust for positioning
  labs(
    title = "Variance of Function Words",
    x = "Rank of Function Words (Descending Variance)",
    y = "Variance"
  ) +
  theme_minimal()
#print(variances)


# Assuming `human_all_words` and `gptm_all_words` are your original datasets
# Top 5 function word indices (from your output)
top3_word_indices <- c(52, 39, 58)
top6_word_indices <- c(52, 39, 58, 5, 1,27)
top15_word_indices <- c(52, 39, 58, 5, 1,27,51, 29, 30, 61, 8, 19,7, 25, 10)
topic70_word_indices <- top_words
# Extract columns corresponding to the top 5 function words
human_top3_words <- human_all_words[, top3_word_indices, drop = FALSE]
gptm_top3_words <- gptm_all_words[, top3_word_indices, drop = FALSE]
human_top6_words <- human_all_words[, top6_word_indices, drop = FALSE]
gptm_top6_words <- gptm_all_words[, top6_word_indices, drop = FALSE]
human_top15_words <- human_all_words[, top15_word_indices, drop = FALSE]
gptm_top15_words <- gptm_all_words[, top15_word_indices, drop = FALSE]
human_top70_words <- human_all_words[, topic70_word_indices, drop = FALSE]
gptm_top70_words <- gptm_all_words[, topic70_word_indices, drop = FALSE]

# Combine the extracted features into a list while keeping the structure
combined_top3_words <- list(human = human_top3_words, GPTM = gptm_top3_words)
combined_top6_words <- list(human = human_top6_words, GPTM = gptm_top6_words)
combined_top15_words <- list(human = human_top15_words, GPTM = gptm_top15_words)
combined_top70_words <- list(human = human_top70_words, GPTM = gptm_top70_words)

# Print the structure of the new dataset to verify
print((combined_top5_words))
results_top3 <- evaluate_models(trainset = combined_top3_words, testset = combined_top3_words)
results_top6 <- evaluate_models(trainset = combined_top6_words, testset = combined_top6_words)
results_top15 <- evaluate_models(trainset = combined_top15_words, testset = combined_top15_words)
results_top70 <- evaluate_models(trainset = combined_top70_words, testset = combined_top70_words)
#results_all <- evaluate_models(trainset = combined_q1$features , testset = combined_q1$features)
print(results_top3)
print(results_top6)
print(results_top15)
print(results_top70)
#print(results_all)

# Data preparation
num_function_words <- c(3, 6, 15, 70)  # Corresponding to top3, top6, top15, and all
DA_accuracy <- c(0.7372244, 0.8120118, 0.9185439, 0.9689353)
KNN_accuracy <- c(0.7266434, 0.5524525, 0.7669061, 0.5803236)
RF_accuracy <- c(0.7335468, 0.8235155, 0.9716939, 0.9993093)

# Create a data frame
accuracy_data <- data.frame(
  Function_Words = num_function_words,
  DA = DA_accuracy,
  KNN = KNN_accuracy,
  RF = RF_accuracy
)

# Reshape data for ggplot2
library(reshape2)
accuracy_melted <- melt(accuracy_data, id.vars = "Function_Words", 
                        variable.name = "Method", value.name = "Accuracy")

# Plot the line graph
library(ggplot2)
ggplot(accuracy_melted, aes(x = Function_Words, y = Accuracy, color = Method, group = Method)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = num_function_words) +
  labs(
    title = "Accuracy vs. Number of Function Words",
    x = "Number of Function Words with top variance",
    y = "Accuracy"
  ) +
  theme_minimal()
results_all <- evaluate_models(trainset = combined_q1$features, testset = combined_q1$features)
print(results_all)



simple_evaluate_model <- function(trainset, testset, n_folds = 5) {
  # Set up cross-validation parameters
  set.seed(1)  # Ensure reproducibility
  
  # Initialize accuracy lists
  DAaccuracy_list <- numeric(n_folds)
  KNNaccuracy_list <- numeric(n_folds)
  RFaccuracy_list <- numeric(n_folds)
  
  # Create folds for cross-validation
  folds_human <- sample(rep(1:n_folds, length.out = nrow(trainset$human)))
  folds_gptm <- sample(rep(1:n_folds, length.out = nrow(trainset$GPTM)))
  
  # Perform cross-validation
  for (fold in 1:n_folds) {
    # Training and testing split for human samples
    train_human <- trainset$human[folds_human != fold, , drop = FALSE]
    test_human <- testset$human[folds_human == fold, , drop = FALSE]
    
    # Training and testing split for GPTM samples
    train_gptm <- trainset$GPTM[folds_gptm != fold, , drop = FALSE]
    test_gptm <- testset$GPTM[folds_gptm == fold, , drop = FALSE]
    
    # Combine training and test sets
    train_fold <- list(train_human, train_gptm)
    test_fold <- rbind(test_human, test_gptm)
    
    # Create ground truth for the test set
    truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))
    train_labels <- c(rep(1, nrow(train_human)), rep(2, nrow(train_gptm)))
    
    
    # Train and predict using the models
    predsDA_fold <- discriminantCorpus(train_fold, test_fold)
    predsKNN_fold <- KNNCorpus(train_fold, test_fold)
    predsRF_fold <- randomForestCorpus(train_fold, test_fold)
    
    # Calculate accuracy for each model
    DAaccuracy_list[fold] <- sum(predsDA_fold == truth_fold) / length(truth_fold)
    KNNaccuracy_list[fold] <- sum(predsKNN_fold == truth_fold) / length(truth_fold)
    RFaccuracy_list[fold] <- sum(predsRF_fold == truth_fold) / length(truth_fold)
  }
  
  # Calculate mean accuracy across all folds
  DA_mean_accuracy <- mean(DAaccuracy_list)
  KNN_mean_accuracy <- mean(KNNaccuracy_list)
  RF_mean_accuracy <- mean(RFaccuracy_list)
  
  # Return the accuracies
  return(list(
    DA_accuracy = DA_mean_accuracy,
    KNN_accuracy = KNN_mean_accuracy,
    RF_accuracy = RF_mean_accuracy
  ))
}
# Example usage of the
