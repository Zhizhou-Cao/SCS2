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
    y = "Variance") +
  theme_minimal()

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
kable(Q4_accuracy_table)

# Convert the data frame to long format for ggplot
Q4accuracy_long <- reshape2::melt(Q4_accuracy_table, 
                                id.vars = "num_function_words", 
                                variable.name = "Method", 
                                value.name = "Accuracy")



# Plot
ggplot(Q4accuracy_long, aes(x = num_function_words, y = Accuracy, color = Method)) +
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
    text = element_text(size = 12),
    legend.position = c(0.75, 0.5),    # Center the legend
    legend.background = element_rect(fill = "white", color = "grey"), # Optional: Add a border
    legend.title = element_text(face = "bold"),
    legend.key.size = unit(0.8, "cm")  # Adjust legend key size
  )

results_all <- evaluate_model(combined_q1$features, combined_q1$features)
kable(results_all)











