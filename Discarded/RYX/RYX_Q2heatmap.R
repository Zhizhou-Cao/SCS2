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



### Heat Map Plot -----
# Define row and column labels
rownames(DA_matrix) <- c("Train-500", "Train-750", "Train-1000")
colnames(DA_matrix) <- c("Test-500", "Test-750", "Test-1000")

rownames(KNN_matrix) <- c("Train-500", "Train-750", "Train-1000")
colnames(KNN_matrix) <- c("Test-500", "Test-750", "Test-1000")

rownames(RF_matrix) <- c("Train-500", "Train-750", "Train-1000")
colnames(RF_matrix) <- c("Test-500", "Test-750", "Test-1000")

# Install and load required packages
if (!require("gridExtra")) install.packages("gridExtra")
if (!require("pheatmap")) install.packages("pheatmap")
library(gridExtra)
library(pheatmap)

# Define separate color gradients for each heatmap
DA_colors <- colorRampPalette(c("white", "blue"))(100)
KNN_colors <- colorRampPalette(c("white", "green"))(100)
RF_colors <- colorRampPalette(c("white", "red"))(100)

# Function to dynamically set text color based on value
set_text_color <- function(matrix, threshold = 0.7) {
  # Values above threshold get white text, otherwise black text
  apply(matrix, c(1, 2), function(x) ifelse(x > threshold, "white", "black"))
}

# Generate individual heatmaps with text color dynamically adjusted
DA_plot <- pheatmap(DA_matrix, cluster_rows = FALSE, cluster_cols = FALSE,
                    main = "DA Accuracy Matrix",
                    color = DA_colors,
                    display_numbers = TRUE, number_color = set_text_color(DA_matrix, 0.96),  # Dynamic text color
                    number_format = "%.4f", angle_col = 0)

KNN_plot <- pheatmap(KNN_matrix, cluster_rows = FALSE, cluster_cols = FALSE,
                     main = "KNN Accuracy Matrix",
                     color = KNN_colors,
                     display_numbers = TRUE, number_color = set_text_color(KNN_matrix, 0.65),  # Dynamic text color
                     number_format = "%.4f", angle_col = 0)

RF_plot <- pheatmap(RF_matrix, cluster_rows = FALSE, cluster_cols = FALSE,
                    main = "RF Accuracy Matrix",
                    color = RF_colors,
                    display_numbers = TRUE, number_color = set_text_color(RF_matrix, 0.97),  # Dynamic text color
                    number_format = "%.4f", angle_col = 0)

# Create a blank plot for layout balance
blank_plot <- grid::textGrob("Combined Accuracy Heatmaps", gp = grid::gpar(fontsize = 16))

# Arrange the plots into a 2x2 layout
grid.arrange(DA_plot$gtable, KNN_plot$gtable,
             RF_plot$gtable, blank_plot,
             ncol = 2, nrow = 2)



### Q3

