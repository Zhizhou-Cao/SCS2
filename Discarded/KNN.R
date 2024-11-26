KNNCorpus <- function(traindata, testdata, k=3) {
  train <- NULL
  trainlabels <- numeric()
  for (i in 1:length(traindata)) {
   # train <- rbind(train, apply(traindata[[i]],2,sum))
    train <- rbind(train, traindata[[i]])
    trainlabels <- c(trainlabels, rep(i,nrow(traindata[[i]])))
  }
  
  for (i in 1:nrow(train)) {
    train[i,] <- train[i,]/sum(train[i,])
  }
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }
  #trainlabels <- 1:nrow(train)
  myKNN(train, testdata, trainlabels,k=k)
}
myKNN <- function(traindata, testdata, trainlabels, k=3) {
  if (mode(traindata) == 'numeric' && !is.matrix(traindata)) {
    traindata <- matrix(traindata,nrow=1)
  }
  if (mode(testdata) == 'numeric' && !is.matrix(testdata)) {
    testdata <- matrix(testdata,nrow=1)
  }
  
  mus <- apply(traindata,2,mean) 
  sigmas <- apply(traindata,2,sd)
  
  ###
  # Store sigmas in a list
  sigmas_list <- as.list(sigmas)
  
  # Count the number of 0s in sigmas
  zero_count <- sum(sigmas == 0)
  ###
  
  for (i in 1:ncol(traindata)) {
    traindata[,i] <- (traindata[,i] - mus[i])/sigmas[i]
  }
  
  for (i in 1:ncol(testdata)) {
    testdata[,i] <- (testdata[,i]-mus[i])/sigmas[i]
  }
  
  preds <- knn(traindata, testdata, trainlabels, k)
  return(preds)
}



# Load required data
train_random <- combined_q1$features

# Set up cross-validation parameters
set.seed(1) # Ensure reproducibility of folds
n_folds <- 5
# Initialize accuracy lists for each method
KNNaccuracy_list <- numeric(n_folds)
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
  
  # Debugging: Check the size of the training sets
  print(paste("Train human size: ", nrow(train_human)))
  print(paste("Train GPTM size: ", nrow(train_gptm)))
  
  
  # Combine training sets
  train_fold <- rbind(train_human, train_gptm)
  train_labels <- c(rep(1, nrow(train_human)), rep(2, nrow(train_gptm)))
  
  # Combine test sets
  test_fold <- rbind(test_human, test_gptm)
  truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))
  
  # Train and predict using KNN
  predsKNN_fold <- KNNCorpus(list(train_human, train_gptm), test_fold, k=7)
  KNNaccuracy_list[fold] <- sum(predsKNN_fold == truth_fold) / length(truth_fold)
}

# Calculate average accuracy for all methods across folds

KNNmean_accuracy <- mean(KNNaccuracy_list)
KNNmean_accuracy





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
print(paste("Optimal k:", optimal_k))
print(paste("Accuracy with optimal k:", optimal_k_accuracy))

# Create a data frame for plotting
accuracy_df <- data.frame(
  k = k_values,
  mean_accuracy = mean_accuracies
)

# Generate the ggplot
ggplot(accuracy_df, aes(x = k, y = mean_accuracy)) +
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
           vjust = -1, hjust = 0.5, color = "red")










