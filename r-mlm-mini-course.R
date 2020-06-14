rm(list = ls())

getwd()

setwd(paste(getwd(), "/", "r-mlm-mini-course", sep = ""))

install.packages("caret"); install.packages("klaR")

library("caret"); library("klaR"); install.packages("caretEnsemble")

download.file(url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", destfile = "iris.csv")

dataset <- read.csv("iris.csv")

attach(dataset)

colnames(dataset) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")

head(dataset)

summary(dataset)

pairs(iris)


#6
# calculate the pre-process parameters from the dataset 
preprocessParams <- preProcess(dataset[ ,1:4], method = c("range"))

# transform the dataset using the pre-processing parameters 
transformed <- predict(preprocessParams, dataset[ ,1:4])

# summarize the transformed dataset 
summary(transformed)


#7
# define training control
trainControl <- trainControl(method = "cv", number = 10) 

# estimate the accuracy of Naive Bayes on the dataset
fit <- train(Species~., data = dataset, trControl = trainControl, method = "nb") 

# summarize the estimated accuracy 
print(fit)


#8 #9 #10
# prepare 5-fold cross-validation and keep the class probabilities
control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = mnLogLoss) 

# estimate accuracy using LogLoss of the CART algorithm
fit <- train(Species~., data = dataset, method = "rpart", metric = "logLoss", trControl = control) 

# display results 
print(fit)


#10
# load the Pima Indians Diabetes dataset 
data(PimaIndiansDiabetes)

# prepare 10-fold cross-validation
trainControl <- trainControl(method = "cv", number = 10) 

# estimate accuracy of logistic regression
set.seed(7)
fit.lr <- train(diabetes~., data = PimaIndiansDiabetes, method = "glm", trControl = trainControl) 

# estimate accuracy of linear discriminate analysis
set.seed(7)
fit.lda <- train(diabetes~., data = PimaIndiansDiabetes, method = "lda", trControl = trainControl) 

# collect resampling statistics
results <- resamples(list(LR = fit.lr, LDA = fit.lda)) 

# summarize results
summary(results)

# plot the results
dotplot(results)


#11
# define training control
trainControl <- trainControl(method = "cv", number = 10) 

# define a grid of parameters to search for random forest 
grid <- expand.grid(.mtry = c(1, 2, 3, 4, 5, 6, 7, 8, 10)) 

# estimate the accuracy of Random Forest on the dataset
fit <- train(Species~., data = dataset, trControl = trainControl, tuneGrid = grid, method = "rf") 

# summarize the estimated accuracy 
print(fit)


#12
# Load packages
install.packages("caretEnsemble")

library(mlbench); library(caret); library(caretEnsemble)

# load the Pima Indians Diabetes dataset 
data(PimaIndiansDiabetes)

# create sub-models
trainControl <- trainControl(method = "cv", number = 5, savePredictions = TRUE, classProbs = TRUE) 


algorithmList <- c("knn", "glm")

set.seed(7)
models <- caretList(diabetes~., data = PimaIndiansDiabetes, trControl=trainControl, methodList = algorithmList)

print(models) 

# learn how to best combine the predictions
stackControl <- trainControl(method="cv", number=5, savePredictions=TRUE, classProbs=TRUE) 

set.seed(7)
stack.glm <- caretStack(models, method="glm", trControl=stackControl) 

print(stack.glm)
