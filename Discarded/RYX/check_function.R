# Import Libraries
library(ggplot2)
library(lattice)
library(class)
library(caret)
library(knitr)
library(randomForest)
source("stylometryfunctions.R") 
source("reducewords.R")

humanM <- loadCorpus("functionwords/functionwords/humanfunctionwords/","functionwords")
GPTM <- loadCorpus("functionwords/functionwords/GPTfunctionwords/","functionwords")

humanM$authornames <- rep(0, length(humanM$authornames))
GPTM$authornames <- rep(1, length(GPTM$authornames))


# Q1 ----
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
# Random Forest
predsRF_random_accuracy <- sum(predsRF_random==truth_random)/length(truth_random)
predsRF_random_accuracy



# K-Fold 

# Set up parameters
set.seed(1) # Ensure reproducibility
fold_range <- 3:10 # Range of folds to test
accuracy_table <- data.frame(
  n_folds = fold_range,
  DA_accuracy = numeric(length(fold_range)),
  KNN_accuracy = numeric(length(fold_range)),
  RF_accuracy = numeric(length(fold_range))
)

# Loop through each fold count (from 5 to 10)
for (f in 1:length(fold_range)) {
  n_folds <- fold_range[f]
  
  # Initialize accuracy lists for each method
  DAaccuracy_list <- numeric(n_folds)
  KNNaccuracy_list <- numeric(n_folds)
  RFaccuracy_list <- numeric(n_folds)
  
  # Create folds for each class (human and GPTM)
  folds_human <- sample(rep(1:n_folds, length.out = nrow(train_random$human)))
  folds_gptm <- sample(rep(1:n_folds, length.out = nrow(train_random$GPTM)))
  
  # Perform cross-validation
  for (fold in 1:n_folds) {
    # Split data for human samples
    train_human <- train_random$human[folds_human != fold, , drop = FALSE]
    test_human <- train_random$human[folds_human == fold, , drop = FALSE]
    
    # Split data for GPTM samples
    train_gptm <- train_random$GPTM[folds_gptm != fold, , drop = FALSE]
    test_gptm <- train_random$GPTM[folds_gptm == fold, , drop = FALSE]
    
    # Combine training and test sets
    train_fold <- list(train_human, train_gptm)
    test_fold <- rbind(test_human, test_gptm)
    
    # Create ground truth for the test set
    truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))
    
    # Train and predict using different methods
    predsDA_fold <- discriminantCorpus(train_fold, test_fold)
    predsKNN_fold <- KNNCorpus(train_fold, test_fold)
    predsRF_fold <- randomForestCorpus(train_fold, test_fold)
    
    # Calculate accuracy for the fold
    DAaccuracy_list[fold] <- sum(predsDA_fold == truth_fold) / length(truth_fold)
    KNNaccuracy_list[fold] <- sum(predsKNN_fold == truth_fold) / length(truth_fold)
    RFaccuracy_list[fold] <- sum(predsRF_fold == truth_fold) / length(truth_fold)
  }
  
  # Store mean accuracy for the current fold count
  accuracy_table$DA_accuracy[f] <- mean(DAaccuracy_list)
  accuracy_table$KNN_accuracy[f] <- mean(KNNaccuracy_list)
  accuracy_table$RF_accuracy[f] <- mean(RFaccuracy_list)
}

# Print the accuracy table
print(accuracy_table)



# Plotting the accuracy table using ggplot2
accuracy_long <- tidyr::pivot_longer(
  accuracy_table,
  cols = c("DA_accuracy", "KNN_accuracy", "RF_accuracy"),
  names_to = "Method",
  values_to = "Accuracy"
)

# Rename methods for better readability
accuracy_long$Method <- factor(
  accuracy_long$Method,
  levels = c("DA_accuracy", "KNN_accuracy", "RF_accuracy"),
  labels = c("Discriminant Analysis", "KNN", "Random Forest")
)

# Create the plot

ggplot(accuracy_long, aes(x = n_folds, y = Accuracy, color = Method)) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 3, alpha = 0.6) +
  labs(
    title = "Accuracy of Different Models across Varying Number of Folds",
    x = "Number of Folds",
    y = "Accuracy",
    color = "Method"
  ) +
  scale_x_continuous(breaks = fold_range) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.title = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    legend.position = c(0.5, 0.5), # Updated position argument
    legend.background = element_rect(fill = "white", color = "grey", linewidth = 0.5) # Updated linewidth argument
  )





# Q3----

# Model 1 train: 纯"Architecture"; test: "Architecture"
humanM$authornames <- rep(0, length(humanM$authornames))
GPTM$authornames <- rep(1, length(GPTM$authornames))

humanM_Q31 <- humanM
GPTM_Q31 <- GPTM
# Identify the index of the "Architecture" folder in booknames
Architecture_index <- 97
if (length(Architecture_index) > 0) {
  humanM_Q31$booknames <- humanM_Q31$booknames[Architecture_index]
  humanM_Q31$features <- humanM_Q31$features[Architecture_index]
  humanM_Q31$authornames <- humanM_Q31$authornames[Architecture_index]
  
  GPTM_Q31$booknames <- GPTM_Q31$booknames[Architecture_index]
  GPTM_Q31$features <- GPTM_Q31$features[Architecture_index]
  GPTM_Q31$authornames <- GPTM_Q31$authornames[Architecture_index]}

# Create a new list for Q3.1
Architecture <- list(
  features = list(
    human = humanM_Q31$features,
    GPTM = GPTM_Q31$features),
  authornames = c("human", "GPT"))

Architecture$features$human <- do.call(rbind, Architecture$features$human)
Architecture$features$GPTM <- do.call(rbind, Architecture$features$GPTM)


# ----check sd是否为0-----

# 定义函数：计算每个位置的标准差
compute_sd_per_position <- function(features_list) {
  # 将所有矩阵按行堆叠成一个大矩阵
  combined_matrix <- do.call(rbind, features_list)
  
  # 按列计算标准差
  apply(combined_matrix, 2, sd)
}

# 计算 GPT 和 Human 中每个位置的标准差
gpt_sd <- compute_sd_per_position(GPTM_Q32$features[Architecture_index])
human_sd <- compute_sd_per_position(humanM_Q32$features[Architecture_index])

# 打印结果
cat("GPT SD:", paste(round(gpt_sd, 2), collapse = " "), "\n")
cat("Human SD:", paste(round(human_sd, 2), collapse = " "), "\n")

# ---------
  
  
# 简单版k-fold
# Load required data
train_random <- Architecture$features

# Set up cross-validation parameters
set.seed(1) # Ensure reproducibility of folds
n_folds <- 5
# Store accuracy for each fold
DAaccuracy_list <- numeric(n_folds) 
KNNaccuracy_list <- numeric(n_folds)
RFaccuracy_list <- numeric(n_folds)
# Create folds for each class (human and GPTM)
folds_human <- sample(rep(1:n_folds, length.out = nrow(train_random$human)))
folds_gptm <- sample(rep(1:n_folds, length.out = nrow(train_random$GPTM)))

# Perform 5-fold cross-validation
for (fold in 1:n_folds) {
  # Training and testing data split for human samples
  train_human <- train_random$human[folds_human != fold, , drop = FALSE]
  test_human <- train_random$human[folds_human == fold, , drop = FALSE]
  
  # Training and testing data split for GPTM samples
  train_gptm <- train_random$GPTM[folds_gptm != fold, , drop = FALSE]
  test_gptm <- train_random$GPTM[folds_gptm == fold, , drop = FALSE]
  
  # Combine training and test sets
  train_fold <- list(train_human, train_gptm)
  test_fold <- rbind(test_human, test_gptm)
  
  # Create ground truth for the test set
  truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))
  
  # Train and predict using discriminant analysis
  predsDA_fold <- discriminantCorpus(train_fold, test_fold)
  predsKNN_fold <- KNNCorpus(train_fold, test_fold)
  predsRF_fold <- randomForestCorpus(train_fold, test_fold)
  
  # Calculate accuracy for the fold
  DAaccuracy_list[fold] <- sum(predsDA_fold == truth_fold) / length(truth_fold)
  KNNaccuracy_list[fold] <- sum(predsKNN_fold == truth_fold) / length(truth_fold)
  RFaccuracy_list[fold] <- sum(predsRF_fold == truth_fold) / length(truth_fold)
}

# Average accuracy across all folds
DAmean_accuracy <- mean(DAaccuracy_list)
DAmean_accuracy
KNNmean_accuracy <- mean(KNNaccuracy_list)
KNNmean_accuracy
RFmean_accuracy <- mean(RFaccuracy_list)
RFmean_accuracy












### -------------ryx 找sd--------------
# Model 2 train: 全部剩下 + "Architecture"; test: "Architecture"
humanM$authornames <- rep(0, length(humanM$authornames))
GPTM$authornames <- rep(1, length(GPTM$authornames))

# Identify the index of the "Architecture" folder in booknames
Architecture_index <- which(sapply(humanM $booknames, function(x) any(grepl("Architecture", x))))
Stories_index <- which(sapply(humanM$booknames, function(x) any(grepl("Stories and literature", x))))

# Define a function to compute the standard deviation for each position 找sd
compute_sd_per_position <- function(features_list) {
  # Combine all matrices in the list by stacking them row-wise into a single matrix
  combined_matrix <- do.call(rbind, features_list)
  
  # Compute the standard deviation for each column (position)
  apply(combined_matrix, 2, sd)
}

# Calculate the standard deviation for each position in GPT and Human data
gpt_sd_Arch <- compute_sd_per_position(GPTM$features[Architecture_index])
human_sd_Arch <- compute_sd_per_position(humanM$features[Architecture_index])

gpt_sd_Story <- compute_sd_per_position(GPTM$features[Stories_index])
human_sd_Story <- compute_sd_per_position(humanM$features[Stories_index])

cat("GPT SD for Architecture:", paste(round(gpt_sd_Arch, 2), collapse = " "), "\n")
cat("Human SD for Architecture:", paste(round(human_sd_Arch, 2), collapse = " "), "\n")

cat("GPT SD for Stories:", paste(round(gpt_sd_Story, 2), collapse = " "), "\n")
cat("Human SD for Stories:", paste(round(human_sd_Story, 2), collapse = " "), "\n")

### ------


# Create a new list for Q3.2
WithoutArchitecture <- list(
  features = list(
    humanM = humanM_Q32$features,
    GPTM = GPTM_Q32$features),
  authornames = c("human", "GPT"))

WithoutArchitecture$features$GPTM <- do.call(rbind, WithoutArchitecture$features$GPTM)
WithoutArchitecture$features$humanM <- do.call(rbind, WithoutArchitecture$features$humanM)

# Train and predict using discriminant analysis
predsDA_fold <- discriminantCorpus(train_fold, test_fold)
predsKNN_fold <- KNNCorpus(train_fold, test_fold)
predsRF_fold <- randomForestCorpus(train_fold, test_fold)













# 简单版k-fold
# Load required data
train_random <- combined_q1$features

# Set up cross-validation parameters
set.seed(1) # Ensure reproducibility of folds
n_folds <- 5
# Store accuracy for each fold
DAaccuracy_list <- numeric(n_folds) 
KNNaccuracy_list <- numeric(n_folds)
RFaccuracy_list <- numeric(n_folds)
# Create folds for each class (human and GPTM)
folds_human <- sample(rep(1:n_folds, length.out = nrow(train_random$human)))
folds_gptm <- sample(rep(1:n_folds, length.out = nrow(train_random$GPTM)))

# Perform 5-fold cross-validation
for (fold in 1:n_folds) {
  # Training and testing data split for human samples
  train_human <- train_random$human[folds_human != fold, , drop = FALSE]
  test_human <- train_random$human[folds_human == fold, , drop = FALSE]
  
  # Training and testing data split for GPTM samples
  train_gptm <- train_random$GPTM[folds_gptm != fold, , drop = FALSE]
  test_gptm <- train_random$GPTM[folds_gptm == fold, , drop = FALSE]
  
  # Combine training and test sets
  train_fold <- list(train_human, train_gptm)
  test_fold <- rbind(test_human, test_gptm)
  
  # Create ground truth for the test set
  truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))
  
  # Train and predict using discriminant analysis
  predsDA_fold <- discriminantCorpus(train_fold, test_fold)
  predsKNN_fold <- KNNCorpus(train_fold, test_fold)
  predsRF_fold <- randomForestCorpus(train_fold, test_fold)
  # Calculate accuracy for the fold
  DAaccuracy_list[fold] <- sum(predsDA_fold == truth_fold) / length(truth_fold)
  KNNaccuracy_list[fold] <- sum(predsKNN_fold == truth_fold) / length(truth_fold)
  RFaccuracy_list[fold] <- sum(predsRF_fold == truth_fold) / length(truth_fold)
}

# Average accuracy across all folds
DAmean_accuracy <- mean(DAaccuracy_list)
DAmean_accuracy
KNNmean_accuracy <- mean(KNNaccuracy_list)
KNNmean_accuracy
RFmean_accuracy <- mean(RFaccuracy_list)
RFmean_accuracy


