---
title: "Machine Learning Final Assignment"
author: "Arunay Kumar"
date: "29 May 2016"
---
## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self-movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

__Data__:

* [Training Data for the assignment is available here:](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) 
* [Test Data for the assignment is available here:](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)


## References:
Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6

* [Read more:](http://groupware.les.inf.puc-rio.br/har#ixzz48MdnjbXS)

### Execution Environment

* Training and Testing data is downloaded for the assignment under the following directory

I am using my own working directory please define your own working directory for code to execute


```r
library(rattle)
library(randomForest)
library(caret)
library(rpart)
library(rpart.plot)

set.seed(65432)

wd<-"f:/DataScience/Coursera-DataScienceSpecialization/08_PracticalMachineLearning/00assignment/"
setwd(wd)
```

* Loading Training and Testing Data for Analysis


```r
training <- read.csv(paste(wd,"pml-training.csv",sep=""))
testing  <- read.csv(paste(wd,"pml-testing.csv",sep=""))
```

* Preprocessing

Step One, we see the data for obvious errors and remove the glaring deviations


```r
str(training)
```

We observe training set variables contains number of factor variables, besides this, number of variables have value "#DIV/0!", We remove all variables which have Near Zero Variaance or are NA.


```r
# remove variables with Nearly Zero Variance
removeVars <- nearZeroVar(training)
training <- training[, -removeVars]
testing  <- testing[, -removeVars]
```

After Loading training Set We divide training Set into Test and Train Set in ratio of 70% and 30%


```r
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainSet <- training[inTrain, ]
TestSet  <- training[-inTrain, ]
dim(TrainSet)
```

```
## [1] 13737   100
```

In addition to this we also remove variables which are all NULL


```r
# remove variables that are mostly NA
allNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[,allNA==FALSE]
TestSet  <- TestSet[, allNA==FALSE]
dim(TrainSet)
```

In Training Data Set Variables X, user_name, raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp varaibles are not predictor variables therefore we ignore them from out analysis



```r
# remove variables which are not predictor Variables 
TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]
dim(TrainSet)
```

```
## [1] 13737    95
```
Model Building

Both Test set and training set is properly pre processed and is ready for analysis. I will apply 3 models Random Forest, Decision Tree and Boosted Model and select the best on training set to apply to quiz questions

As basic Preprocessing step we standardize all the training set data and preserve it for use on testing Data set

#### a) Random Forest 

```r
set.seed(65432)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modelRF  <- train(classe ~ ., data=TrainSet, method="rf", trControl=controlRF)
modelRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 19.06%
## Confusion matrix:
##    A  B  C  D  E class.error
## A 69  2  3  1  0   0.0800000
## B  6 36  5  2  1   0.2800000
## C  5  2 46  0  0   0.1320755
## D  5  1  4 29  5   0.3409091
## E  2  5  2  2 45   0.1964286
```

Prediction on Test Set


```r
predictRF  <- predict(modelRF, newdata=TestSet)
ConfRF     <- confusionMatrix(predictRF, TestSet$classe)
```

```
## Error in table(data, reference, dnn = dnn, ...): all arguments must have the same length
```

```r
ConfRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B    1 1136    4    0    0
##          C    0    1 1022    2    0
##          D    0    1    0  962    4
##          E    0    0    0    0 1078
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9976         
##                  95% CI : (0.996, 0.9987)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.997          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9974   0.9961   0.9979   0.9963
## Specificity            0.9998   0.9989   0.9994   0.9990   1.0000
## Pos Pred Value         0.9994   0.9956   0.9971   0.9948   1.0000
## Neg Pred Value         0.9998   0.9994   0.9992   0.9996   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1930   0.1737   0.1635   0.1832
## Detection Prevalence   0.2845   0.1939   0.1742   0.1643   0.1832
## Balanced Accuracy      0.9996   0.9982   0.9977   0.9985   0.9982
```

#### b) Decision Tree


```r
set.seed(65432)
modelDT <- rpart(classe ~ ., data=TrainSet, method="class")
fancyRpartPlot(modelDT)
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-1.png)

#### Prediction on Test Set


```r
predictDT <- predict(modelDT, newdata=TestSet, type="class")
confDT <- confusionMatrix(predictDT, TestSet$classe)
confDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1483  218   22   86   51
##          B   58  624   43   30   29
##          C   10   57  829  128   55
##          D  102  189   77  616  141
##          E   21   51   55  104  806
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7405          
##                  95% CI : (0.7291, 0.7517)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.671           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8859   0.5478   0.8080   0.6390   0.7449
## Specificity            0.9105   0.9663   0.9485   0.8966   0.9519
## Pos Pred Value         0.7973   0.7959   0.7683   0.5476   0.7772
## Neg Pred Value         0.9525   0.8990   0.9590   0.9269   0.9431
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2520   0.1060   0.1409   0.1047   0.1370
## Detection Prevalence   0.3161   0.1332   0.1833   0.1912   0.1762
## Balanced Accuracy      0.8982   0.7571   0.8783   0.7678   0.8484
```

#### c) Boosting

```r
set.seed(65432)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modelGBM   <- train(classe ~ ., data=TrainSet, method = "gbm", trControl = controlGBM, verbose = FALSE)
modelGBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 100 iterations were performed.
## There were 94 predictors of which 75 had non-zero influence.
```
#### Prediction on Test Set

```r
# prediction on Test dataset
predictGBM <- predict(modelGBM, newdata=TestSet)
confGBM <- confusionMatrix(predictGBM, TestSet$classe)
```

```
## Error in table(data, reference, dnn = dnn, ...): all arguments must have the same length
```

```r
confGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1670    8    0    0    0
##          B    2 1116    6    1    9
##          C    1   14 1013    4    2
##          D    1    1    6  959   18
##          E    0    0    1    0 1053
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9874          
##                  95% CI : (0.9842, 0.9901)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9841          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9798   0.9873   0.9948   0.9732
## Specificity            0.9981   0.9962   0.9957   0.9947   0.9998
## Pos Pred Value         0.9952   0.9841   0.9797   0.9736   0.9991
## Neg Pred Value         0.9990   0.9952   0.9973   0.9990   0.9940
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2838   0.1896   0.1721   0.1630   0.1789
## Detection Prevalence   0.2851   0.1927   0.1757   0.1674   0.1791
## Balanced Accuracy      0.9979   0.9880   0.9915   0.9948   0.9865
```

## Applying the Selected Model to the Test Data

Of the three classification techniques used, given below the accuracy of the three methods 

Random Forest : 0.9976
Decision Tree : 0.7183
GBM           : 0.9874

From the data we infere that Random forest performs optimum and we will use this for out prediction 


```r
predictTesting <- predict(modelRF, newdata=testing)
```

```
## Error: variables 'max_roll_belt', 'max_picth_belt', 'min_roll_belt', 'min_pitch_belt', 'amplitude_roll_belt', 'amplitude_pitch_belt', 'var_total_accel_belt', 'avg_roll_belt', 'stddev_roll_belt', 'var_roll_belt', 'avg_pitch_belt', 'stddev_pitch_belt', 'var_pitch_belt', 'avg_yaw_belt', 'stddev_yaw_belt', 'var_yaw_belt', 'var_accel_arm', 'max_picth_arm', 'max_yaw_arm', 'min_yaw_arm', 'amplitude_yaw_arm', 'max_roll_dumbbell', 'max_picth_dumbbell', 'min_roll_dumbbell', 'min_pitch_dumbbell', 'amplitude_roll_dumbbell', 'amplitude_pitch_dumbbell', 'var_accel_dumbbell', 'avg_roll_dumbbell', 'stddev_roll_dumbbell', 'var_roll_dumbbell', 'avg_pitch_dumbbell', 'stddev_pitch_dumbbell', 'var_pitch_dumbbell', 'avg_yaw_dumbbell', 'stddev_yaw_dumbbell', 'var_yaw_dumbbell', 'max_picth_forearm', 'min_pitch_forearm', 'amplitude_pitch_forearm', 'var_accel_forearm' were specified with different types from the fit
```

```r
predictTesting
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
