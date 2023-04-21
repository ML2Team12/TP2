library(data.table)
library(feather)
library(glmnet)
library(mice)
library(miceadds)
library(caret)
library(gbm)
install.packages("e1071")
setwd("D:/Netease")
# 读取feather文件
d <- read_feather("jane_street_train.feather")

d <- as.data.frame(d)

# we only use a subset of it, given the size of the dataset, to make it available to run locally

data<-d[sample(1:nrow(d),0.01*nrow(d)), ]

rm(d)
gc()

# The column "resp" represents the rate of return over a certain period of time. 
# If "resp" is greater than 0,
# it means that the investment portfolio may have generated a positive return and vice versa for negative returns. 
# By setting a threshold (e.g. labeling cases where "resp" is greater than 0 as "action" equals 1), 
# we can transform the problem into a binary classification problem.
# Weight stand for how important this particular investment is in portfolio

# When weight is equal to 0, it means that the trade should not be included in the overall statistics, 
# as these trades may be invalid or should not be considered. 


#resp_1:resp_4,penalty terms in this specific competition, to penalize over fit

# delete rows that have weight of 0
data <- data[data$weight != 0, ]

# create 'action' column
data$action <- as.integer(data$resp > 0)


# subsetting dataframe to get X

x_feature<-subset(data,select = -c(weight,action,resp,resp_1,resp_2,resp_3,resp_4,ts_id))


miss_count <- colSums(is.na(x_feature))
miss_vars <- sum(miss_count > 0)
miss_vars
# 88 columns contain missing values


x_feature_imputed <- mice(x_feature, method = "pmm", m = 1,maxit=1)
# "PMM" stands for "predictive mean matching" a method for imputing missing values in the MICE package. 
# The method involves first predicting the missing values using regression or other methods,
# and then finding the closest matching observed value from the dataset to replace the missing value. 
x_feature_imputed<-complete(x_feature_imputed)


pc <- prcomp(x_feature_imputed,
             center = TRUE,
             scale = TRUE)
summary(pc)
pca_features<-pc$x[,1:50]
pca_features

## train test split ####

trainIndex<-sample(1:nrow(pca_features),0.7*nrow(pca_features))
pca_train_x<-pca_features[trainIndex,]
pca_test_x<-pca_features[-trainIndex,]
y_train<-data$action[trainIndex]
y_test<-data$action[-trainIndex]


##### first model, simple logistic regression model
train<-as.data.frame(cbind(data$action[trainIndex],pca_train_x))
test <- as.data.frame(cbind(data$action[-trainIndex], pca_test_x))
train$V1 <- as.factor(train$V1)
test$V1 <- as.factor(test$V1)
logreg_model <- glm(V1 ~ ., data = train, family = "binomial") #y_train is action
predicted_probs <- predict(logreg_model, test, type = "response")
predicted_class <- ifelse(predicted_probs > 0.5, 1, 0)


# Confusion matrix
cm <- confusionMatrix(as.factor(predicted_class), as.factor(y_test))
print(cm)
# Accuracy : 0.5205



##### second model, logistic regression model with elastic net

# Set up a grid of lambda values for cross-validation
lambda_grid <- 10^seq(10, -2, length.out = 5)

# Fit the logistic regression model using glmnet with alpha = 0 for Ridge regression
logreg_model_glmnet <- cv.glmnet(pca_train_x, y_train, family = "binomial", alpha = 0.5, lambda = lambda_grid)

# Get the best lambda value
best_lambda <- logreg_model_glmnet$lambda.min

# Train the final model with the best lambda value
logreg_elasticnet <- glmnet(pca_train_x, y_train, family = "binomial", alpha = 0.5, lambda = best_lambda)

# Predict on the test dataset
predicted_probs <- predict(logreg_elasticnet, pca_test_x, type = "response")
predicted_class <- ifelse(predicted_probs > 0.5, 1, 0)

cm <- confusionMatrix(as.factor(predicted_class), as.factor(y_test))
print(cm)

# Accuracy : 0.5129  


### model 3 gradient boost
gbm_model <- gbm(
  V1~.,
  data = train,
  n.trees = 1000,
  interaction.depth = 8,
  shrinkage = 0.01,
  cv.folds = 3,
  distribution = "bernoulli"
)

gbm_pred <- predict(gbm_model, newdata = test, n.trees = 1000, type = "response")
gbm_pred_class<-ifelse(gbm_pred > 0.5, 1, 0)
cm <- confusionMatrix(as.factor(gbm_pred_class), as.factor(y_test))
print(cm)


###support vector machine ###
library(e1071)

# Train SVM model
svm_model <- svm(V1 ~ ., data = train, kernel = "radial", cost = 0.1, scale = TRUE)

# Predict on the test dataset
svm_predicted_class <- predict(svm_model, test)
svm_predicted_class
# Confusion matrix

cm_svm <- confusionMatrix(as.factor(svm_predicted_class), as.factor(y_test))
print(cm_svm)

d2<-cbind(data$action,x_feature_imputed)
colnames(d2)[1] ="action"
trainIndex2<-sample(1:nrow(d2),0.7*nrow(d2))
train2<-d2[trainIndex2,]
test2<-d2[-trainIndex2,]
train2$action <- as.factor(train2$action)
test2$action <- as.factor(test2$action)

svm_model2 <- svm(action ~ ., data = train2, kernel = "radial", cost = 0.1, scale = TRUE)
svm_predicted_class2 <- predict(svm_model2, test2)

cm_svm <- confusionMatrix(as.factor(svm_predicted_class2), as.factor(test2$action),positive="1")
print(cm_svm)

#### regression based on resp

train<-as.data.frame(cbind(data$resp[trainIndex],pca_train_x))
test <- as.data.frame(cbind(data$resp[-trainIndex], pca_test_x))
lm_model <- lm(V1 ~ ., data = train) #y_train is action
predicted_probs <- predict(lm_model, newdata=test)
predicted_class <- ifelse(predicted_probs > 0, 1, 0)
predicted_class
cm <- confusionMatrix(as.factor(predicted_class), as.factor(data$action[-trainIndex]))
print(cm)

