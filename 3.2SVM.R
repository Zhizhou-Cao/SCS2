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

humanM_Q32 <- humanM
GPTM_Q32 <- GPTM
if (length(StoryLit_index) > 0) {
  humanM_Q32$features <- humanM_Q32$features[-StoryLit_index]
  GPTM_Q32$features <- GPTM_Q32$features[-StoryLit_index]}

# Create a new list for Q3.3
Without_StoryLit <- list(
  features = list(
    human = humanM_Q32$features,
    GPTM = GPTM_Q32$features),
  authornames = c("human", "GPT"))
Without_StoryLit$features$human <- do.call(rbind, Without_StoryLit$features$human)
Without_StoryLit$features$GPTM <- do.call(rbind, Without_StoryLit$features$GPTM)


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


train_svm(Without_StoryLit$features,StoryLit$features)




# Q3.2 ----
# Used trained model(without 'StoryLit') to test StoryLit
topic <- 97 #Stories and literature
humanfeatures_story <- humanM$features[[topic]] #select the essays on this particular topic
GPTfeatures_story <- GPTM$features[[topic]]

features_only_story <- rbind(humanfeatures, GPTfeatures) #this is a matrix of both human and GPT essays#authornames_story <- c(rep(0,nrow(humanfeatures)), rep(1,nrow(GPTfeatures)))

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
predsRF_all <- randomForestCorpus(combined_q1$features, test_data)

# Evaluate Accuracy
DA_without_story_accuracy <- sum(predsDA_without_story == truth_test) / length(truth_test)
DA_all_accuracy <- sum(predsDA_all == truth_test) / length(truth_test)

KNN_without_story_accuracy <- sum(predsKNN_without_story == truth_test) / length(truth_test)
KNN_all_accuracy <- sum(predsKNN_all == truth_test) / length(truth_test)

RF_without_story_accuracy <- sum(predsRF_without_story == truth_test) / length(truth_test)
RF_all_accuracy <- sum(predsRF_all == truth_test) / length(truth_test)

SVM_without_story_accuracy <- train_svm(combined_without_story$features, combined_only_story$features)
SVM_all_accuracy <- train_svm(combined_q1$features, combined_only_story$features)

# Print Results
cat("Discriminant Analysis Accuracy without 'story' in the model:", DA_without_story_accuracy, "\n")
cat("K-Nearest Neighbors Accuracy without 'story' in the model:", KNN_without_story_accuracy, "\n")
cat("Random Forest Accuracy without 'story' in the model:", RF_without_story_accuracy, "\n")
cat("Discriminant Analysis Accuracy with 'story' in the model:", DA_all_accuracy, "\n")
cat("K-Nearest Neighbors Accuracy with 'story' in the model:", KNN_all_accuracy, "\n")
cat("Random Forest Accuracy with 'story' in the model:", RF_all_accuracy, "\n")













