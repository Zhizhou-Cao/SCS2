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


# leave-one cross validation
train_LOOCV <- combined_q1$features
features <- combined_q1$features

test_LOOCV <- NULL
truth_LOOCV <- NULL
predsDA_LOOCV <- NULL
predsKNN_LOOCV <- NULL
predsRF_LOOCV <- NULL

for (i in 1:length(train_LOOCV)) {
  for (j in 1:nrow(train_LOOCV[[i]])) {
    test_LOOCV <- matrix(features[[i]][j,],nrow=1)
    train_LOOCV <- features
    train_LOOCV[[i]] <- train_LOOCV[[i]][-j,,drop=FALSE]
    
    pred <- discriminantCorpus(train_LOOCV, test_LOOCV)
    predsDA_LOOCV <- c(predsDA_LOOCV, pred)
    
    pred <- KNNCorpus(train_LOOCV, test_LOOCV)
    predsKNN_LOOCV <- c(predsKNN_LOOCV, pred)
    
    pred <- randomForestCorpus(train_LOOCV, test_LOOCV)
    predsRF_LOOCV <- c(predsRF_LOOCV, pred)
    
    truth_LOOCV <- c(truth_LOOCV, i)
  }}

# Multinomial (more than two categories) discriminant analysis
DA_LOOCV_accuracy <- sum(predsDA_LOOCV==truth_LOOCV)/length(truth_LOOCV)
DA_LOOCV_accuracy
# KNN
KNN_LOOCV_accuracy <-sum(predsKNN_LOOCV==truth_LOOCV)/length(truth_LOOCV)
KNN_LOOCV_accuracy
# RF
RF_LOOCV_accuracy <-sum(predsRF_LOOCV==truth_LOOCV)/length(truth_LOOCV)

RF_LOOCV_accuracy


# Present the accuracy in table
accuracy_table <- data.frame(
  "One random CV" = c(DA_random_accuracy, KNN_random_accuracy),
  "LOOCV" = c(DA_LOOCV_accuracy, KNN_LOOCV_accuracy),
  row.names = c("Discriminant Analysis", "K-Nearest Neighbors"))
kable(accuracy_table)




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
Architecture_index <- which(sapply(humanM_Q31$booknames, function(x) any(grepl("Architecture", x))))
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













# Model 2 train: 全部剩下 + "Architecture"; test: "Architecture"
humanM$authornames <- rep(0, length(humanM$authornames))
GPTM$authornames <- rep(1, length(GPTM$authornames))

humanM_Q32 <- humanM
GPTM_Q32 <- GPTM
# Identify the index of the "Architecture" folder in booknames
Architecture_index <- which(sapply(humanM_Q32$booknames, function(x) any(grepl("Architecture", x))))
if (length(Architecture_index) > 0) {
  humanM_Q32$booknames <- humanM_Q32$booknames[-Architecture_index]
  humanM_Q32$features <- humanM_Q32$features[-Architecture_index]
  humanM_Q32$authornames <- humanM_Q32$authornames[-Architecture_index]
  
  GPTM_Q32$booknames <- GPTM_Q32$booknames[-Architecture_index]
  GPTM_Q32$features <- GPTM_Q32$features[-Architecture_index]
  GPTM_Q32$authornames <- GPTM_Q32$authornames[-Architecture_index]}

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













#不可用----

# 合并列表
combined_features <- c(humanM$features, GPTM$features)
combined_authornames <- c(humanM$authornames, GPTM$authornames)
combined_booknames <- c(humanM$booknames, GPTM$booknames)
combined_list <- list(features = combined_features, 
                      authornames = combined_authornames, 
                      booknames = combined_booknames)

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



# Random select cross validation
train_random <- combined_list$features
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

truth_random1 <- unlist(lapply(truth_random, function(x) ifelse(x >= 0 & x <= 110, 0, 1)))
predsDA_random1 <- unlist(lapply(predsDA_random, function(x) ifelse(x >= 0 & x <= 110, 0, 1)))
predsKNN_random1 <- unlist(lapply(predsKNN_random, function(x) ifelse(x >= 0 & x <= 110, 0, 1)))
# Multinomial (more than two categories) discriminant analysis
DA_random_accuracy <- sum(predsDA_random1==truth_random1)/length(truth_random1)
DA_random_accuracy
# KNN 
KNN_random_accuracy <- sum(predsKNN_random1==truth_random1)/length(truth_random1)
KNN_random_accuracy



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


