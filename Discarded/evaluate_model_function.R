evaluate_model <- function(trainset, testset, fold_range = 3:10) {
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

trainset <- combined_q1$features
testset <- combined_q1$features
n_folds <- 3:5
results <- evaluate_model(trainset, testset, n_folds)
print(results)








