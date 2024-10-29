
reducewords <- function(features, numwords) {
  newfeatures <- features
  numfeatures <- ncol(features)
  
  for (i in 1:nrow(features)) {
    prob <- features[i,]/sum(features[i,])
    temp <- sample(1:numfeatures, numwords,replace=TRUE,prob=prob)
    counts <- numeric(numfeatures)
    for (j in 1:length(temp)) {
      counts[temp[j]] <- counts[temp[j]] + 1
    }
    newfeatures[i,] <- counts
  }
  return(newfeatures)
}