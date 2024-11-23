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

# SVM --
# Train SVM model
svm_model <- svm(Without_StoryLit$features, as.factor(train_labels), kernel ="linear")

# Predict using SVM
svm_preds <- predict(svm_model, Without_StoryLit$features)


# Training and testing data split for human samples
train_human <- Without_StoryLit$features$human
test_human <- StoryLit$features$human

# Training and testing data split for GPTM samples
train_gptm <- Without_StoryLit$features$GPTM
test_gptm <- StoryLit$features$GPTM

# Combine training sets
train_fold <- rbind(train_human, train_gptm)
train_labels <- c(rep(1, nrow(train_human)), rep(2, nrow(train_gptm)))

# Combine test sets
test_fold <- rbind(test_human, test_gptm)
truth_fold <- c(rep(1, nrow(test_human)), rep(2, nrow(test_gptm)))

# Train SVM model
svm_model <- svm(train_fold, as.factor(train_labels), kernel ="linear")

# Predict using SVM
svm_preds <- predict(svm_model, test_fold)

accuracy <- sum(as.numeric(svm_preds) == truth_fold) / length(truth_fold)

