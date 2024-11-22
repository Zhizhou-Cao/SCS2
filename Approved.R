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

# Evaluated function
evaluate_model <- function(trainset, testset, fold_range = 5) {
  # Initialize accuracy table
  accuracy_table <- data.frame(
    n_folds = fold_range,
    DA_accuracy = numeric(length(fold_range)),
    KNN_accuracy = numeric(length(fold_range)),
    RF_accuracy = numeric(length(fold_range)),
    SVM_accuracy = numeric(length(fold_range))
  )
  
  for (f in 1:length(fold_range)) {
    # Current number of folds
    n_folds <- fold_range[f]
    
    # Ensure reproducibility
    set.seed(1)
    
    # Initialize accuracy lists for each method
    DAaccuracy_list <- numeric(n_folds)
    KNNaccuracy_list <- numeric(n_folds)
    RFaccuracy_list <- numeric(n_folds)
    SVMaccuracy_list <- numeric(n_folds)
    
    # Create folds for each class
    folds_human <- sample(rep(1:n_folds, length.out = nrow(trainset$human)))
    folds_gptm <- sample(rep(1:n_folds, length.out = nrow(trainset$GPTM)))
    
    # Perform cross-validation
    for (fold in 1:n_folds) {
      # Split data for human samples
      train_human <- trainset$human[folds_human != fold, , drop = FALSE]
      test_human <- testset$human[folds_human == fold, , drop = FALSE]
      
      # Split data for GPTM samples
      train_gptm <- trainset$GPTM[folds_gptm != fold, , drop = FALSE]
      test_gptm <- testset$GPTM[folds_gptm == fold, , drop = FALSE]
      
      # Combine training sets and labels
      train_fold <- rbind(train_human, train_gptm)
      train_labels <- c(rep(1, nrow(train_human)), rep(2, nrow(train_gptm)))
      
      # Combine test sets
      test_fold <- rbind(test_human, test_gptm)
      truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))
      
      # Train and predict using different methods
      predsDA_fold <- discriminantCorpus(list(train_human, train_gptm), test_fold)
      predsKNN_fold <- KNNCorpus(list(train_human, train_gptm), test_fold)
      predsRF_fold <- randomForestCorpus(list(train_human, train_gptm), test_fold)
      
      # Train and predict using SVM
      svm_model <- svm(train_fold, as.factor(train_labels), kernel = "linear", probability = TRUE)
      svm_preds <- predict(svm_model, test_fold)
      
      # Calculate accuracy for each method
      DAaccuracy_list[fold] <- sum(predsDA_fold == truth_fold) / length(truth_fold)
      KNNaccuracy_list[fold] <- sum(predsKNN_fold == truth_fold) / length(truth_fold)
      RFaccuracy_list[fold] <- sum(predsRF_fold == truth_fold) / length(truth_fold)
      SVMaccuracy_list[fold] <- sum(as.numeric(svm_preds) == truth_fold) / length(truth_fold)
    }
    
    # Store mean accuracy for the current fold count
    accuracy_table$DA_accuracy[f] <- mean(DAaccuracy_list)
    accuracy_table$KNN_accuracy[f] <- mean(KNNaccuracy_list)
    accuracy_table$RF_accuracy[f] <- mean(RFaccuracy_list)
    accuracy_table$SVM_accuracy[f] <- mean(SVMaccuracy_list)
  }
  
  return(accuracy_table)
}



# Q1 ----
# Create a new list for Q1
combined_q1 <- list(
  features = list(
    human = humanM$features,
    GPTM = GPTM$features),
  authornames = c(0, 1))

combined_q1$features$GPTM <- do.call(rbind, combined_q1$features$GPTM)
combined_q1$features$human <- do.call(rbind, combined_q1$features$human)

# K-Fold iteration for four methods

# Set up parameters
fold_range <- 3:10 # Range of folds to test
results <- evaluate_model(combined_q1$features, combined_q1$features, fold_range)

# Print the accuracy table
kable(results, caption = "Average Accuracy Across Methods", format = "markdown")

# Visualisation
# Plotting the accuracy table using ggplot2
accuracy_long <- tidyr::pivot_longer(
  results,
  cols = c("DA_accuracy", "KNN_accuracy", "RF_accuracy", "SVM_accuracy"),
  names_to = "Method",
  values_to = "Accuracy"
)

# Rename methods for better readability
accuracy_long$Method <- factor(
  accuracy_long$Method,
  levels = c("DA_accuracy", "KNN_accuracy", "RF_accuracy", "SVM_accuracy"),
  labels = c("Discriminant Analysis", "KNN", "Random Forest", "Support Vector Machine")
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


# Q2 -----
















# Q3 -----

# Q3.1 -----
# Select twp topics(Architecture/ Stories and literature) for modelling

# Stories and Literature
humanM_Q31 <- humanM
GPTM_Q31 <- GPTM
# Identify the index of the "Stories and literature" folder in booknames
StoryLit_index <- which(sapply(humanM_Q31$booknames, function(x) any(grepl("Stories and literature", x))))
if (length(StoryLit_index) > 0) {
  humanM_Q31$features <- humanM_Q31$features[StoryLit_index]
  GPTM_Q31$features <- GPTM_Q31$features[StoryLit_index]}

# Create a new list for Q3.1
StoryLit <- list(
  features = list(
    human = humanM_Q31$features,
    GPTM = GPTM_Q31$features),
  authornames = c("human", "GPT"))
StoryLit$features$human <- do.call(rbind, StoryLit$features$human)
StoryLit$features$GPTM <- do.call(rbind, StoryLit$features$GPTM)

# K-fold CV & 4 Methods (same as Q1)
accuracy_table_3.1 <- evaluate_model(StoryLit$features, StoryLit$features)

# Print the table
kable(accuracy_table_3.1, caption = "Story and literature Average Accuracy Across Methods", format = "markdown")




# Architecture
humanM_Q31 <- humanM
GPTM_Q31 <- GPTM
# Identify the index of the "Architecture" folder in booknames
Architecture_index <- which(sapply(humanM_Q31$booknames, function(x) any(grepl("Architecture", x))))
if (length(Architecture_index) > 0) {
  humanM_Q31$features <- humanM_Q31$features[Architecture_index]
  GPTM_Q31$features <- GPTM_Q31$features[Architecture_index]}

# Create a new list for Q3.1
Architecture <- list(
  features = list(
    human = humanM_Q31$features,
    GPTM = GPTM_Q31$features),
  authornames = c("human", "GPT"))
Architecture$features$human <- do.call(rbind, Architecture$features$human)
Architecture$features$GPTM <- do.call(rbind, Architecture$features$GPTM)

# Exist Error, Explain here!!!!!
evaluate_model(Architecture$features, Architecture$features)












# Q3.2 ----
# Used trained model(without 'StoryLit') to test StoryLit















# Q3.3 -----
# 










# Q4 ------
# Reduced Words













