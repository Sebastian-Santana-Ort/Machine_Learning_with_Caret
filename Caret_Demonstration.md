# Machine Learning Pipelines and Model Building in Caret
Sebastian Santana Ortiz

- [Set up & Data Management](#set-up--data-management)
- [Building pipeline](#building-pipeline)
  - [Splitting data](#splitting-data)
  - [Pipeline Building](#pipeline-building)
- [Building models](#building-models)
- [Testing models](#testing-models)
- [Required Packages](#required-packages)

The purpose of this file is to provide a quick demonstration on how to
set up a machine learning pipeline and build a set of predictive models
using the [caret package](https://topepo.github.io/caret/index.html) in
R.

The text below will be split into three sections:

1.  Set up & Data management
2.  Building a pipeline
3.  Building models
4.  Testing models

Note that the data used for this demonstration is from the [Kaggle
competition](https://www.kaggle.com/competitions/titanic/overview)
“**Titanic - Machine Learning from Disaster”,** a beginner friendly
dataset that is commonly used for demonstration purposes.

## Set up & Data Management

``` r
# Load required packages
library(caret)
library(caretEnsemble)
library(data.table)
library(tidyr)
library(dplyr)
library(kableExtra)
library(skimr)
```

The data provided in the Kaggle challenge is already split for us into a
training and testing set. The next step is to understand the types of
features/factors we have access to in the training set.

``` r
# Load training and testing DFs
train_raw = fread('./Data/train.csv')
test_raw = fread('./Data/test.csv')
```

According to the data set description the following variables should be
present in the training set.

|              |                                             |                                                |
|:------------:|:-------------------------------------------:|:----------------------------------------------:|
| **Variable** |               **Description**               |                    **Key**                     |
|   survival   |                  Survival                   |                0 = No, 1 = Yes                 |
|    pclass    |                Ticket class                 |           1 = 1st, 2 = 2nd, 3 = 3rd            |
|     sex      |                     Sex                     |                                                |
|     Age      |                Age in years                 |                                                |
|    sibsp     | \# of siblings / spouses aboard the Titanic |                                                |
|    parch     | \# of parents / children aboard the Titanic |                                                |
|    ticket    |                Ticket number                |                                                |
|     fare     |               Passenger fare                |                                                |
|    cabin     |                Cabin number                 |                                                |
|   embarked   |             Port of Embarkation             | C = Cherbourg, Q = Queenstown, S = Southampton |

``` r
# Quick summary of training set

kable(summary(train_raw%>%
                select_if(is.character)))
```

|     | Name             | Sex              | Ticket           | Cabin            | Embarked         |
|:----|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
|     | Length:891       | Length:891       | Length:891       | Length:891       | Length:891       |
|     | Class :character | Class :character | Class :character | Class :character | Class :character |
|     | Mode :character  | Mode :character  | Mode :character  | Mode :character  | Mode :character  |

``` r
# Note: we should only look at the testing set once and only once (i.e., when we test the final model)
```

``` r
# Quick summary of training set

kable(summary(train_raw%>%
                select_if(~! is.character(.))))
```

|     | PassengerId   | Survived       | Pclass        | Age           | SibSp         | Parch          | Fare           |
|:----|:--------------|:---------------|:--------------|:--------------|:--------------|:---------------|:---------------|
|     | Min. : 1.0    | Min. :0.0000   | Min. :1.000   | Min. : 0.42   | Min. :0.000   | Min. :0.0000   | Min. : 0.00    |
|     | 1st Qu.:223.5 | 1st Qu.:0.0000 | 1st Qu.:2.000 | 1st Qu.:20.12 | 1st Qu.:0.000 | 1st Qu.:0.0000 | 1st Qu.: 7.91  |
|     | Median :446.0 | Median :0.0000 | Median :3.000 | Median :28.00 | Median :0.000 | Median :0.0000 | Median : 14.45 |
|     | Mean :446.0   | Mean :0.3838   | Mean :2.309   | Mean :29.70   | Mean :0.523   | Mean :0.3816   | Mean : 32.20   |
|     | 3rd Qu.:668.5 | 3rd Qu.:1.0000 | 3rd Qu.:3.000 | 3rd Qu.:38.00 | 3rd Qu.:1.000 | 3rd Qu.:0.0000 | 3rd Qu.: 31.00 |
|     | Max. :891.0   | Max. :1.0000   | Max. :3.000   | Max. :80.00   | Max. :8.000   | Max. :6.0000   | Max. :512.33   |
|     | NA            | NA             | NA            | NA’s :177     | NA            | NA             | NA             |

``` r
# Note: we should only look at the testing set once and only once (i.e., when we test the final model)
```

Note above that based on the raw data needs to be modified before we use
them to build our models.

- The “pclass” variable needs to be relabeled as *factor*.

  - Survival on the other can be left as a binary variable. In this
    case, our models will be able to predict the probability of survival
    and this can easily be turned into a binary variable later on (e.g.,
    all predictions above .5 are relabeledto 1 or “survived”).

- I will also remove passenger ID, this variable can be brought back
  later on but should not have any predictive capacity.

``` r
# Note that at this point I am only recoding or removing variables. Centering will take place later on.

train = train_raw%>%
  select(-PassengerId)%>%
  mutate(Pclass = as.factor(Pclass))

test = test_raw%>%
  select(-PassengerId)%>%
  mutate(Pclass = as.factor(Pclass))
```

## Building pipeline

Pipelines are necessary tools in machine learning that organize and
standardize sequential transformations to a set of features/factors.
They can ensure that both the training and test sets have the *exact
same operations done*.

It is worth noting that in machine learning, there is a crucial
principle: no information from the testing fold should bleed into our
prediction model. If we were to center our features on the *global
average* (including observations from the training and testing folds)
before splitting the data, then our model would have information on the
testing set even if it only trained on the training observations.

Hence, when we center variables, we should only use values from the
training set. Then, we can apply those transformations to the test set.

### Splitting data

In this specific case, the data is already split for us. However, in
case there would be a need to split the data, the code below would
result in a training data set and test data set.

``` r
# This step is crucial to ensure reporudicbility when splitting data
set.seed(101) 

# This example code is further splitting the train dataset into exampple folds
# Create a 75% and 25% split
intrain = createDataPartition(y=train$Survived, p=0.75, list=FALSE)
example_75_fold = train[intrain,]; train$Survived = as.factor(train$Survived)
example_25_fold = train[-intrain,]; train$Survived = as.factor(train$Survived)
```

### Pipeline Building

In this case, we have both numeric and character variables. Hence, we
will have to apply at least two types of transformation: scaling
(centering all variables around the mean) and dummy coding (binarizing
categorical variables based on unique categories).

In this case we do not have missing data, so there is no need for
imputation. However, if it was needed imputation (or some equivalent)
would be done at this point. Additionally, there are many other forms of
transformations designed to reduce skew, normalize distributions, make
features linearly separable, and more. For this demonstration, I will
not be using this advanced techniques but it is worth acknowledging they
exist.

## Building models

## Testing models

## Required Packages
