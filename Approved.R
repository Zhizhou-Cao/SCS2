
# Import Libraries
library(ggplot2)
library(lattice)
library(class)
library(caret)
library(knitr)
library(e1071) # SVM

source("stylometryfunctions.R")
source("reducewords.R")

humanM <- loadCorpus("functionwords/functionwords/humanfunctionwords/","functionwords")
GPTM <- loadCorpus("functionwords/functionwords/GPTfunctionwords/","functionwords")

humanM$authornames <- rep(0, length(humanM$authornames))
GPTM$authornames <- rep(1, length(GPTM$authornames))

# Evaluate Model----
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
      predsKNN_fold <- KNNCorpus(list(train_human, train_gptm), test_fold, k = 7)
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


# train_SVM-----
train_svm <- function(train_data, test_data) {
  # Extract training and testing data
  train_human <- train_data$human
  train_gptm <- train_data$GPTM
  test_human <- test_data$human
  test_gptm <- test_data$GPTM
  
  # Combine training sets
  train_fold <- rbind(train_human, train_gptm)
  train_labels <- c(rep(1, nrow(train_human)), rep(2, nrow(train_gptm)))
  
  # Combine test sets
  test_fold <- rbind(test_human, test_gptm)
  truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))
  
  # Train SVM model
  svm_model <- svm(train_fold, as.factor(train_labels), kernel = "linear")
  
  # Predict using SVM
  svm_preds <- predict(svm_model, test_fold)
  
  # Calculate accuracy
  accuracy <- sum(as.numeric(svm_preds) == truth_fold) / length(truth_fold)
  
  return(accuracy)
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


# Find Optimal K for KNN
# Load required data
train_random <- combined_q1$features

# Set up cross-validation parameters
set.seed(1) # Ensure reproducibility of folds
n_folds <- 5
k_values <- seq(1, 33, by = 2) # Range of k values to test
mean_accuracies <- numeric(length(k_values)) # Store mean accuracy for each k

# Generate folds for cross-validation
folds_human <- sample(rep(1:n_folds, length.out = nrow(train_random$human)))
folds_gptm <- sample(rep(1:n_folds, length.out = nrow(train_random$GPTM)))

# Function to perform KNN for a given k
perform_knn <- function(k) {
  KNNaccuracy_list <- numeric(n_folds) # Initialize accuracy list for this k
  
  for (fold in 1:n_folds) {
    # Training and testing data split for human samples
    train_human <- train_random$human[folds_human != fold, , drop = FALSE]
    test_human <- train_random$human[folds_human == fold, , drop = FALSE]
    
    # Training and testing data split for GPTM samples
    train_gptm <- train_random$GPTM[folds_gptm != fold, , drop = FALSE]
    test_gptm <- train_random$GPTM[folds_gptm == fold, , drop = FALSE]
    
    # Combine training sets
    train_fold <- rbind(train_human, train_gptm)
    train_labels <- c(rep(1, nrow(train_human)), rep(2, nrow(train_gptm)))
    
    # Combine test sets
    test_fold <- rbind(test_human, test_gptm)
    truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))
    
    # Train and predict using KNN
    predsKNN_fold <- KNNCorpus(list(train_human, train_gptm), test_fold, k = k)
    KNNaccuracy_list[fold] <- sum(predsKNN_fold == truth_fold) / length(truth_fold)
  }
  
  return(mean(KNNaccuracy_list))
}

# Loop over all k values and compute mean accuracy
for (i in seq_along(k_values)) {
  k <- k_values[i]
  mean_accuracies[i] <- perform_knn(k)
}

# Find the optimal k
optimal_k <- k_values[which.max(mean_accuracies)]
optimal_k_accuracy <- max(mean_accuracies)

# Print results
#print(paste("Optimal k:", optimal_k))
#print(paste("Accuracy with optimal k:", optimal_k_accuracy))

# Create a data frame for plotting
accuracy_df <- data.frame(
  k = k_values,
  mean_accuracy = mean_accuracies
)

# Generate the ggplot
(optimal_K_plot <- ggplot(accuracy_df, aes(x = k, y = mean_accuracy)) +
  geom_line(color = "blue", size = 1) +   # Line for accuracy
  geom_point(color = "blue", size = 3, alpha = 0.5) + # Points for each k
  geom_vline(xintercept = optimal_k, color = "red", linetype = "dashed") + # Optimal k
  labs(
    title = "Optimal k for kNN",
    x = "k (Number of Neighbors)",
    y = "Mean Accuracy"
  ) +
  ylim(0.978, 0.9875)+
  theme_minimal(base_size = 15) + # Clean theme with larger text
  annotate("text", x = optimal_k, y = optimal_k_accuracy, 
           label = paste("Optimal k =", optimal_k), size = 8,
           vjust = -1, hjust = 0.5, color = "red"))

ggsave(filename = "optimal_K_plot.png", plot = optimal_K_plot)





# K-Fold iteration for four methods
# Set up parameters
range_of_fold <- 3:10 # Range of folds to test

accuracy_table <- evaluate_model(combined_q1$features, combined_q1$features, range_of_fold)
 
# Print the accuracy table
kable(accuracy_table, caption = "Average Accuracy Across Methods", format = "latex", booktabs = TRUE)

# Visualisation
# Plotting the accuracy table using ggplot2
accuracy_long <- tidyr::pivot_longer(
  accuracy_table,
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

q1accuracy_plot<- ggplot(accuracy_long, aes(x = n_folds, y = Accuracy, color = Method)) +
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
    plot.title = element_text(hjust = 0.5, size = 20),
    axis.title = element_text(size = 20),
    #legend.title = element_text(size = 5),
    #legend.text = element_text(size = 7),
    #legend.position = c(0.1, 0.1), # Updated position argument
    #legend.background = element_rect(fill = "white", color = "grey", linewidth = 0.5) # Updated linewidth argument
    )
q1accuracy_plot
ggsave(filename = "q1accuracy_plot.png", plot = q1accuracy_plot)

# Q2 -----
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

results_matrices <- list()

trainsets <- list(combined_500$features, combined_750$features, combined_1000$features)
testsets <- list(combined_500$features, combined_750$features, combined_1000$features)


# Loop over trainsets and testsets
for (i in 1:length(trainsets)) {
  for (j in 1:length(testsets)) {
    # Evaluate the model
    results <- evaluate_model(trainsets[[i]], testsets[[j]])
    
    # Create a descriptive name for the matrix
    matrix_name <- paste0("Train-", c(500, 750, 1000)[i], "_Test-", c(500, 750, 1000)[j])
    
    # Create a matrix from the results
    results_matrix <- data.frame(
      #n_folds = results$n_folds,
      DA_accuracy = results$DA_accuracy,
      KNN_accuracy = results$KNN_accuracy,
      RF_accuracy = results$RF_accuracy,
      SVM_accuracy = results$SVM_accuracy
    )
    
    # Store the matrix in the list with the descriptive name
    results_matrices[[matrix_name]] <- results_matrix
  }
}

# Print the results
print(results_matrices)

# Initialize an empty data frame
final_table <- data.frame(
  Trainset = character(),
  Testset = character(),
  DA_accuracy = numeric(),
  KNN_accuracy = numeric(),
  RF_accuracy = numeric(),
  SVM_accuracy = numeric(),
  stringsAsFactors = FALSE
)

# Loop through the results_matrices to extract data
for (name in names(results_matrices)) {
  # Extract trainset and testset names from the list name
  split_name <- strsplit(name, "_")[[1]]
  trainset <- gsub("Train-", "", split_name[1])
  testset <- gsub("Test-", "", split_name[2])
  
  # Extract the accuracy values from the matrix
  accuracy_values <- results_matrices[[name]]
  
  # Append the row to the final table
  final_table <- rbind(
    final_table,
    data.frame(
      Trainset = trainset,
      Testset = testset,
      DA_accuracy = accuracy_values$DA_accuracy,
      KNN_accuracy = accuracy_values$KNN_accuracy,
      RF_accuracy = accuracy_values$RF_accuracy,
      SVM_accuracy = accuracy_values$SVM_accuracy
    )
  )
}

# Print the final table
print(final_table)
kable(final_table, format = 'latex',booktabs = TRUE, caption = "Accuracy with Full 71 Function Words")

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
# Load required data
train_random <- StoryLit$features
# Set up cross-validation parameters
set.seed(1) # Ensure reproducibility of folds
n_folds <- 5 # KNN has greatest accuracy

# Initialize accuracy lists for each method
DAaccuracy_list <- numeric(n_folds)
KNNaccuracy_list <- numeric(n_folds)
RFaccuracy_list <- numeric(n_folds)
SVMaccuracy_list <- numeric(n_folds) # For SVM

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
  
  # Combine training sets
  train_fold <- rbind(train_human, train_gptm)
  train_labels <- c(rep(1, nrow(train_human)), rep(2, nrow(train_gptm)))
  
  # Combine test sets
  test_fold <- rbind(test_human, test_gptm)
  truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))
  
  # Train and predict using discriminant analysis
  predsDA_fold <- discriminantCorpus(list(train_human, train_gptm), test_fold)
  DAaccuracy_list[fold] <- sum(predsDA_fold == truth_fold) / length(truth_fold)
  
  # Train and predict using KNN
  predsKNN_fold <- KNNCorpus(list(train_human, train_gptm), test_fold)
  KNNaccuracy_list[fold] <- sum(predsKNN_fold == truth_fold) / length(truth_fold)
  
  # Train and predict using Random Forest
  predsRF_fold <- randomForestCorpus(list(train_human, train_gptm), test_fold)
  RFaccuracy_list[fold] <- sum(predsRF_fold == truth_fold) / length(truth_fold)
  
  # Train and predict using SVM
  svm_model <- svm(train_fold, as.factor(train_labels), kernel = "linear", probability = TRUE)
  svm_preds <- predict(svm_model, test_fold)
  SVMaccuracy_list[fold] <- sum(as.numeric(svm_preds) == truth_fold) / length(truth_fold)
}

# Calculate average accuracy for all methods across folds
DAmean_accuracy <- mean(DAaccuracy_list)
KNNmean_accuracy <- mean(KNNaccuracy_list)
RFmean_accuracy <- mean(RFaccuracy_list)
SVMmean_accuracy <- mean(SVMaccuracy_list)

# Create a table of the results
accuracy_table_3.1 <- data.frame(
  Method = c("DA", "KNN", "Random Forest", "SVM"),
  Average_Accuracy = c(DAmean_accuracy, KNNmean_accuracy, RFmean_accuracy, SVMmean_accuracy)
)

# Print the table
kable(accuracy_table_3.1, caption = "Average Accuracy Across Methods", format = "markdown")




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



# Exist Error, Explain here

humanM_Q312 <- humanM
GPTM_Q312 <- GPTM

# Identify the index of the four folders in booknames
Architecture_index <- which(sapply(humanM $booknames, function(x) any(grepl("Architecture", x))))
Addiction_index <- which(sapply(humanM$booknames, function(x) any(grepl("Addiction", x))))
Stories_index <- which(sapply(humanM$booknames, function(x) any(grepl("Stories and literature", x))))
Art_index <- which(sapply(humanM$booknames, function(x) any(grepl("Art", x))))

# Create a list to store results
results <- list()

# Define the index names and corresponding keywords
indices <- list(
  Architecture = "Architecture",
  Addiction = "Addiction",
  Stories = "Stories and literature",
  Art = "Art"
)

# Initialize an empty data frame to store results
results_table <- data.frame(
  Topic = character(),
  Fold_1_SD_Zeros = numeric(),
  Fold_2_SD_Zeros = numeric(),
  Fold_3_SD_Zeros = numeric(),
  Fold_4_SD_Zeros = numeric(),
  Fold_5_SD_Zeros = numeric(),
  Mean_KNN_Accuracy = numeric(),
  stringsAsFactors = FALSE
)

# Iterate through each index and process it
for (index_name in names(indices)) {
  tryCatch({
    keyword <- indices[[index_name]]
    
    # Generate index based on the current keyword
    current_index <- which(sapply(humanM$booknames, function(x) any(grepl(keyword, x))))
    
    # If matches are found, filter features based on the current index
    if (length(current_index) > 0) {
      humanM_Q312$features <- humanM_Q312$features[current_index]
      GPTM_Q312$features <- GPTM_Q312$features[current_index]
    } else {
      # Skip this iteration if no matches
      next
    }
    
    # Create a new list for the current category
    CurrentCategory <- list(
      features = list(
        human = humanM_Q312$features,
        GPTM = GPTM_Q312$features
      ),
      authornames = c("human", "GPT")
    )
    
    # Combine all feature data for both human and GPTM
    CurrentCategory$features$human <- do.call(rbind, CurrentCategory$features$human)
    CurrentCategory$features$GPTM <- do.call(rbind, CurrentCategory$features$GPTM)
    
    # Assign the features for cross-validation
    train_random <- CurrentCategory$features
    
    # Set up cross-validation parameters
    set.seed(1)
    n_folds <- 5
    
    # Initialize accuracy lists and SD zero counts
    KNNaccuracy_list <- numeric(n_folds) # Accuracy for k-Nearest Neighbors
    SD_zero_counts <- numeric(n_folds)  # Number of zeros in standard deviation
    
    # Create cross-validation folds for each class (human and GPTM)
    folds_human <- sample(rep(1:n_folds, length.out = nrow(train_random$human)))
    folds_gptm <- sample(rep(1:n_folds, length.out = nrow(train_random$GPTM)))
    
    # Perform 5-fold cross-validation
    for (fold in 1:n_folds) {
      tryCatch({
        # Split human data into training and testing sets for this fold
        train_human <- train_random$human[folds_human != fold, , drop = FALSE]
        test_human <- train_random$human[folds_human == fold, , drop = FALSE]
        
        # Split GPTM data into training and testing sets for this fold
        train_gptm <- train_random$GPTM[folds_gptm != fold, , drop = FALSE]
        test_gptm <- train_random$GPTM[folds_gptm == fold, , drop = FALSE]
        
        # Combine training sets from human and GPTM
        train_fold <- rbind(train_human, train_gptm)
        train_labels <- c(rep(1, nrow(train_human)), rep(2, nrow(train_gptm)))
        
        # Combine test sets from human and GPTM
        test_fold <- rbind(test_human, test_gptm)
        truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))
        
        # Compute standard deviation for the current fold and count zeros
        combined_data <- rbind(train_human, train_gptm)
        SD_values <- apply(combined_data, 2, sd)
        SD_zero_counts[fold] <- sum(SD_values == 0)  # Count zeros in SD
        
        # Train and predict using k-Nearest Neighbors
        predsKNN_fold <- KNNCorpus(list(train_human, train_gptm), test_fold)
        KNNaccuracy_list[fold] <- sum(predsKNN_fold == truth_fold) / length(truth_fold)
      }, error = function(e) {
        # If an error occurs, suppress error message and set values to NA
        SD_zero_counts[fold] <- NA
        KNNaccuracy_list[fold] <- NA
      })
    }
    
    # Calculate mean accuracy for KNN
    if (0 %in% KNNaccuracy_list) {
      KNNmean_accuracy <- NA
    } else {
      KNNmean_accuracy <- mean(KNNaccuracy_list, na.rm = TRUE)
    }
    
    # Save results to the table
    results_table <- rbind(results_table, data.frame(
      Topic = index_name,
      Fold_1_SD_Zeros = SD_zero_counts[1],
      Fold_2_SD_Zeros = SD_zero_counts[2],
      Fold_3_SD_Zeros = SD_zero_counts[3],
      Fold_4_SD_Zeros = SD_zero_counts[4],
      Fold_5_SD_Zeros = SD_zero_counts[5],
      Mean_KNN_Accuracy = round(KNNmean_accuracy, 4)
    ))
    
  }, error = function(e) {
    # Suppress the error message for the topic and continue
    return(NULL)
    
  })
  
  # Reset humanM_Q312 and GPTM_Q312 to their original values after each topic
  humanM_Q312 <- humanM
  GPTM_Q312 <- GPTM
}

# Print the final results table
kable(results_table, format = "markdown")









# Q3.2 ----
# Used trained model(without 'StoryLit') to test StoryLit
topic <- 97 #Stories and literature
humanfeatures_story <- humanM$features[[topic]] #select the essays on this particular topic
GPTfeatures_story <- GPTM$features[[topic]]

features_only_story <- rbind(humanfeatures, GPTfeatures) #this is a matrix of both human and GPT essays
#authornames_story <- c(rep(0,nrow(humanfeatures)), rep(1,nrow(GPTfeatures)))

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
predsDA_without_story <- discriminantCorpus(train_data, test_data)
predsDA_all <- discriminantCorpus(combined_q1$features, test_data)

# Perform K-Nearest Neighbors (KNN)
predsKNN_without_story <- KNNCorpus(train_data, test_data)
predsKNN_all <- KNNCorpus(combined_q1$features, test_data)

# Perform Random Forest (RF)
predsRF_without_story <- randomForestCorpus(train_data, test_data)
#predsRF_all <- randomForestCorpus(combined_q1$features, test_data)

# Evaluate Accuracy
DA_without_story_accuracy <- sum(predsDA_without_story == truth_test) / length(truth_test)
#DA_all_accuracy <- sum(predsDA_all == truth_test) / length(truth_test)

KNN_without_story_accuracy <- sum(predsKNN_without_story == truth_test) / length(truth_test)
#KNN_all_accuracy <- sum(predsKNN_all == truth_test) / length(truth_test)

RF_without_story_accuracy <- sum(predsRF_without_story == truth_test) / length(truth_test)
#RF_all_accuracy <- sum(predsRF_all == truth_test) / length(truth_test)

# 新的SVM!!!!!!!!!!------
SVM_without_story_accuracy <- train_svm(combined_without_story$features, combined_only_story$features)
#SVM_all_accuracy <- train_svm(combined_q1$features, combined_only_story$features)

# Print Results
cat("Discriminant Analysis Accuracy without 'story' in the model:", DA_without_story_accuracy, "\n")
cat("K-Nearest Neighbors Accuracy without 'story' in the model:", KNN_without_story_accuracy, "\n")
cat("Random Forest Accuracy without 'story' in the model:", RF_without_story_accuracy, "\n")
# cat("Discriminant Analysis Accuracy with 'story' in the model:", DA_all_accuracy, "\n")
# cat("K-Nearest Neighbors Accuracy with 'story' in the model:", KNN_all_accuracy, "\n")
# cat("Random Forest Accuracy with 'story' in the model:", RF_all_accuracy, "\n")



# Q3.3 -----
# Number of papers and sets
num_papers <- 73
num_sets <- 5
# Divide papers into 5 sets (approximately equal)
# Set seed for reproducibility
# Randomly assign essays to sets
set_indices <- sample(rep(1:num_sets, length.out = num_papers))

# Choose one set as the test set
test_set_id <- 2  # You can change this to evaluate other test sets
test_indices <- which(set_indices == test_set_id)

# Indices for the remaining sets
remaining_sets <- setdiff(1:num_sets, test_set_id)

# Methods to evaluate
methods <- c("DA", "KNN", "RF", "SVM")

# Initialize matrix to store accuracies
accuracy_matrix <- matrix(0, nrow = length(remaining_sets), ncol = length(methods),
                          dimnames = list(paste0("TrainSets=", 1:length(remaining_sets)), methods))

# Test data (fixed throughout the process)
test_human <- combined_only_story$features$human[test_indices, , drop = FALSE]
test_gptm <- combined_only_story$features$GPTM[test_indices, , drop = FALSE]
test_data <- rbind(test_human, test_gptm)
truth_test <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))

# Train data without "Story and Literature"
train_human <- combined_without_story$features$human
train_gptm <- combined_without_story$features$GPTM

# Loop through 1 to 4 trainsets (incrementally adding "Story and Literature" sets)
for (train_size in 1:length(remaining_sets)) {
  # Select train indices from the remaining sets
  train_set_ids <- remaining_sets[1:train_size]
  additional_train_indices <- which(set_indices %in% train_set_ids)
  
  # Train data (combine without "Story" and selected "Story" sets)
  train_human_with_story <- rbind(train_human, combined_only_story$features$human[additional_train_indices, , drop = FALSE])
  train_gptm_with_story <- rbind(train_gptm, combined_only_story$features$GPTM[additional_train_indices, , drop = FALSE])
  
  train_data <- list(human = train_human_with_story, GPTM = train_gptm_with_story)
  
  # Perform DA, KNN, and RF
  predsDA <- discriminantCorpus(train_data, test_data)
  predsKNN <- KNNCorpus(train_data, test_data)
  predsRF <- randomForestCorpus(train_data, test_data)
  
  
  # Prepare SVM training data
  train_features <- rbind(train_data$human, train_data$GPTM)
  train_labels <- c(rep(1, nrow(train_data$human)), rep(2, nrow(train_data$GPTM)))  # Adjust labels accordingly
  
  # Train and predict using SVM
  svm_model <- svm(train_features, as.factor(train_labels), kernel = "linear", probability = TRUE)
  svm_preds <- predict(svm_model, test_data)
  
  # Calculate and store accuracies
  accuracy_matrix[train_size, "DA"] <- sum(predsDA == truth_test) / length(truth_test)
  accuracy_matrix[train_size, "KNN"] <- sum(predsKNN == truth_test) / length(truth_test)
  accuracy_matrix[train_size, "RF"] <- sum(predsRF == truth_test) / length(truth_test)
  
  accuracy_matrix[train_size, "SVM"] <- sum(svm_preds == truth_test) / length(truth_test)
}

# Print the accuracy matrix
print("Accuracy Matrix by Trainset Sizes:")
print(accuracy_matrix)



# Q4 ------
# Reduced Words

# For human features
human_all_words <- combined_q1$features$human
# For GPTM features
gptm_all_words <- combined_q1$features$GPTM
# Combine all features
combined_all_features <- rbind(human_all_words, gptm_all_words)

# Measure feature variances
variances <- apply(combined_all_features, 2, var)
variances <- variances[-71] # Remove the final term

# Reorder by their variance (from large to small)
words_order <- order(variances, decreasing = TRUE)  
words_order

# Extract the variance values of the top 5 function words
variances_order <- variances[words_order]
# Print the variances
print(variances_order)

variance_df <- data.frame(
  Rank = 1:length(variances),
  Variance = variances_order
)

# Variance visualisation
# Add annotations to the plot
points_to_annotate <- c(3, 6, 15, 70)
q4variance_plot<- ggplot(variance_df, aes(x = Rank, y = Variance)) +
  geom_line() +
  geom_point() +
  geom_text(data = variance_df[points_to_annotate, ], 
            aes(label = Rank), 
            vjust = -0.5, 
            color = "red") +  # Adjust vjust for positioning
  labs(
    title = "Variance of Function Words",
    x = "Rank of Function Words (Descending Variance)",
    y = "Variance") +
  theme_minimal()+
  theme(
    text = element_text(size = 20))

q4variance_plot
ggsave(filename = "q4variance_plot.png", plot = q4variance_plot)

# Test the words effect on modelling
top3_word_indices <- words_order[1:3] #c(52, 39, 58)
top6_word_indices <- words_order[1:6] #c(52, 39, 58, 5, 1,27)
top15_word_indices <- words_order[1:15] #c(52, 39, 58, 5, 1,27,51, 29, 30, 61, 8, 19,7, 25, 10)
topic70_word_indices <- words_order

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


# Results through 5 fold CV
results_top3 <- evaluate_model(trainset = combined_top3_words, testset = combined_top3_words)
results_top6 <- evaluate_model(trainset = combined_top6_words, testset = combined_top6_words)
results_top15 <- evaluate_model(trainset = combined_top15_words, testset = combined_top15_words)
results_top70 <- evaluate_model(trainset = combined_top70_words, testset = combined_top70_words)

results_top3
results_top6
results_top15
results_top70
# Combine the accuracy results into vectors
num_function_words <- c(3, 6, 15, 70)  # Top3, Top6, Top15, Top70

DA <- c(
  results_top3$DA_accuracy[1],
  results_top6$DA_accuracy[1],
  results_top15$DA_accuracy[1],
  results_top70$DA_accuracy[1])

KNN <- c(
  results_top3$KNN_accuracy[1],
  results_top6$KNN_accuracy[1],
  results_top15$KNN_accuracy[1],
  results_top70$KNN_accuracy[1])
RF <- c(
  results_top3$RF_accuracy[1],
  results_top6$RF_accuracy[1],
  results_top15$RF_accuracy[1],
  results_top70$RF_accuracy[1])

SVM<- c(
  results_top3$SVM_accuracy[1],
  results_top6$SVM_accuracy[1],
  results_top15$SVM_accuracy[1],
  results_top70$SVM_accuracy[1])

# Combine all the vectors into a single data frame
Q4_accuracy_table <- data.frame(
  num_function_words,
  DA,KNN,RF,SVM)

# View the resulting data frame
kable(Q4_accuracy_table, format = 'latex', booktabs = TRUE, caption = "Accuracy Across Function Words")

# Convert the data frame to long format for ggplot
Q4accuracy_long <- reshape2::melt(Q4_accuracy_table, 
                                  id.vars = "num_function_words", 
                                  variable.name = "Method", 
                                  value.name = "Accuracy")



# Plot
q4words_plot<- ggplot(Q4accuracy_long, aes(x = num_function_words, y = Accuracy, color = Method)) +
  geom_line() +     # Lines connecting points
  geom_point(size = 3, alpha = 0.5) +    # Points for each method
  labs(
    title = "Accuracy vs Number of Function Words",
    x = "Number of Function Words",
    y = "Accuracy",
    color = "Method") +
  scale_x_continuous(breaks = num_function_words) +
  theme_bw() +         # Minimalistic theme
  theme(
    text = element_text(size = 20),
    legend.position = c(0.75, 0.5),    # Center the legend
    legend.background = element_rect(fill = "white", color = "grey"), # Optional: Add a border
    legend.title = element_text(face = "bold"),
    legend.key.size = unit(0.8, "cm")  # Adjust legend key size
  )
q4words_plot
ggsave(filename = "q4words_plot.png", plot = q4words_plot)

results_all <- evaluate_model(combined_q1$features, combined_q1$features)
kable(results_all, format = 'latex',booktabs = TRUE, caption = "Accuracy with Full 71 Function Words")












