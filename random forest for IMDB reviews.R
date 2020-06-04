library(tidyverse)
library(stringr)
library(tm)
library(SnowballC)
library(caTools)
library(randomForest)
library(e1071)
library(caret)
library(tensorflow)
library(magrittr)
library(ggplot2)
library(kernlab)
library(ROCR)


set.seed(222)
reviews<-read.csv("D:/moviereview/moviereview.csv")
head(reviews)

reviews <- reviews %>%
  mutate(stars = as.numeric(str_sub(stars, start = 1, end = 1))) %>%
  mutate(date = as.Date(str_sub(date, start = 4, end = -1), '%B %d, %Y'))

# Text Cleaning
corpus <- VCorpus(VectorSource(reviews$review))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers) 
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords())
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, stripWhitespace)

# creat Document Term Matrix
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.99) 
dtm

# Build the classification model
dataset <- as.data.frame(as.matrix(dtm))
dataset
feat_cols <- ncol(dataset)
feat_cols
dataset$stars <- reviews$stars
dataset$stars

# Take star rating as a factor
dataset$stars <- factor(dataset$stars, levels = c(1:5), ordered = F)

# Splitting training and testing set
Splitting <- sample.split(dataset$stars, SplitRatio = 0.7)
training <- subset(dataset, Splitting == T)
testing <- subset(dataset, Splitting == F)

# Plot testing set in terms of Rating and number
ggplot(data = testing, aes(x = stars)) + geom_bar(fill="darkblue")

# Random Forest
RF <- randomForest(x = training[1:feat_cols],
                   y = training$stars,
                   ntree = 100)

# Predict the Test Set results
pred_RF <- predict(RF, newdata = testing)

# Create the confusion matrices
cm_RF <- table(testing[, (feat_cols + 1)], pred_RF)

# Calculate accuracy scores
accuracy_RF <- ((cm_RF[[1,1]] + cm_RF[[2,2]] + cm_RF[[3,3]] + cm_RF[[4,4]] + cm_RF[[5,5]]) / nrow(testing))

print('Random Forest')
cm_RF
accuracy_RF

