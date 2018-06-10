# !diagnostics off
rm(list = ls())
library(tidyverse)
# install.packages(pkgs = "caret", 
#                  dependencies = c("Depends", "Imports"))
library(caret)
library(randomForest)
library(purrr)
library(rpart)

# ---- load Kaggle data --------
genderSub <- read.csv("./data/gender_submission.csv")
testDat <- read.csv("./data/test.csv") %>% as.tibble(.)
trainDat <- read.csv("./data/train.csv") %>% as.tibble(.)

# -------- first model - glm ---------
# train the simplest possible model glm
# logistic
trainDat <- trainDat %>% 
  filter(!is.na(Age)) %>% 
  mutate(Pclass_factor = factor(Pclass))

mymod <- glm(Survived ~ Pclass + Sex + Age, data = trainDat, family = "binomial")
mydf <- data.frame(survived_obs = trainDat$Survived, prediction =
                     predict(mymod, type = "response"))
mydf <- mydf %>% 
  mutate(survived_predicted = prediction > 0.5)
my_accuracy <- sum(mydf$survived_predicted == mydf$survived_obs) / nrow(mydf)
my_accuracy
# this is accuracy on training set

# ----- a slightly more complex regression -------
trainDat %>% 
  group_by(Pclass, Survived) %>% 
  summarise(mymean = mean(Fare))
# we can see that the effect of fare is increasing
# with class so an interaction makes sense
# we also notice that there is an interaction between
# Sex and age: the older you are the more of a difference
# being female makes
trainDat %>% 
  mutate(age_cat = case_when(Age <= 1 ~ "baby",
                             Age > 1 & Age <= 5 ~ "toddler",
                             Age > 5 ~ "older")) %>% 
  group_by(age_cat, Sex) %>% 
  summarise(mymean = mean(Survived))

trainDat <- trainDat %>% 
  mutate(age_cat = case_when(Age <= 1 ~ "baby",
                             Age > 1 & Age <= 5 ~ "toddler",
                             Age > 5 ~ "older"))

mymod1 <- glm(Survived ~ Pclass * Fare + age_cat * Sex, 
              data = trainDat, family = "binomial")

testDat <- testDat %>% 
  mutate(age_cat = case_when(Age <= 1 ~ "baby",
                             Age > 1 & Age <= 5 ~ "toddler",
                             Age > 5 ~ "older"))
testDat$age_cat[is.na(testDat$Age)] <- "older"
testDat$Fare[is.na(testDat$Fare)] <- mean(testDat$Fare, na.rm = T)

# get a submission to Kaggle
mypred <- ifelse(predict(mymod1, newdata = testDat, type = "response") > 0.5, 1, 0)
mysub <- tibble(PassengerId = testDat$PassengerId, Survived = mypred)
write.csv(mysub, file = "mysecondprediction.csv", row.names = FALSE)

# -------- regression tree approach --------

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data=trainDat,
             method="class")
plot(fit)
text(fit)
summary(fit)

# new submission to Kaggle
mypred <- predict(fit, testDat, type = "class")
mysub <- tibble(PassengerId = testDat$PassengerId, Survived = mypred)
write.csv(mysub, file = "myfirstprediction.csv", row.names = FALSE)

# --------- random forest ----------
# this first requires removing all NAs
testDat <- testDat %>% mutate(Survived = NA, dataset = "test")
trainDat <- trainDat %>% mutate(dataset = "train")
alldat <- bind_rows(trainDat, testDat)
alldat %>% 
  summarise_all(funs(nb_na = sum(is.na(.))))

# age and fare have some NAs
get_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked,
                 data = alldat,
                 method="anova")
alldat$Age[is.na(alldat$Age)] <-  predict(get_age, alldat %>% filter(is.na(Age)))
alldat$Fare[is.na(alldat$Fare)] <- median(alldat$Fare, na.rm = T)

# also an issue with Embarked
# as there are 2 empty ones.
# replace with "S" as most people
# embarked from there
table(alldat$Embarked)
idx <- which(alldat$Embarked == "")
alldat <- alldat %>% 
  mutate(Embarked = replace(Embarked, idx, "S"))

# if we use survived as integer the function 
# will attempt to run a regression, use factor
# to force it to run a classification
train <- alldat %>% filter(dataset == "train")
train <- train %>% mutate_if(is.character, as.factor)
mytest <- alldat %>% filter(dataset == "test") %>% mutate_if(is.character, as.factor)
# the model can't deal with character variables. 
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                      Embarked, data = train, importance = TRUE, ntree = 2000)


mypred <- predict(fit, mytest)
mysub <- tibble(PassengerId = testDat$PassengerId, Survived = mypred)
write.csv(mysub, file = "myrandomforestprediction.csv", row.names = FALSE)


# try feature engineering

