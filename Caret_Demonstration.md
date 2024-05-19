# Machine Learning Pipelines and Model Building in Caret
Sebastian Santana Ortiz

- [Set up & Data Management](#set-up--data-management)
- [Building pipeline](#building-pipeline)
  - [Splitting data](#splitting-data)
  - [Pipeline Building](#pipeline-building)
- [Building models](#building-models)
- [Testing models](#testing-models)

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
training and testing set. Our first objective is to understand the types
of features/factors we have access to in the training set.

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
# Quick summary of character features training set

skimr::skim(train_raw)%>%
  yank("character")%>%
  select(-complete_rate, -whitespace, -min,-max)
```

**Variable type: character**

| skim_variable | n_missing | empty | n_unique |
|:--------------|----------:|------:|---------:|
| Name          |         0 |     0 |      891 |
| Sex           |         0 |     0 |        2 |
| Ticket        |         0 |     0 |      681 |
| Cabin         |         0 |   687 |      148 |
| Embarked      |         0 |     2 |        4 |

``` r
# Note: we should only look at the testing set once and only once (i.e., when we test the final model)
```

``` r
# Quick summary of all other features in training set

skimr::skim(train_raw)%>%
  yank("numeric")%>%
  select(-hist)
```

**Variable type: numeric**

| skim_variable | n_missing | complete_rate |   mean |     sd |   p0 |    p25 |    p50 |   p75 |   p100 |
|:--------------|----------:|--------------:|-------:|-------:|-----:|-------:|-------:|------:|-------:|
| PassengerId   |         0 |           1.0 | 446.00 | 257.35 | 1.00 | 223.50 | 446.00 | 668.5 | 891.00 |
| Survived      |         0 |           1.0 |   0.38 |   0.49 | 0.00 |   0.00 |   0.00 |   1.0 |   1.00 |
| Pclass        |         0 |           1.0 |   2.31 |   0.84 | 1.00 |   2.00 |   3.00 |   3.0 |   3.00 |
| Age           |       177 |           0.8 |  29.70 |  14.53 | 0.42 |  20.12 |  28.00 |  38.0 |  80.00 |
| SibSp         |         0 |           1.0 |   0.52 |   1.10 | 0.00 |   0.00 |   0.00 |   1.0 |   8.00 |
| Parch         |         0 |           1.0 |   0.38 |   0.81 | 0.00 |   0.00 |   0.00 |   0.0 |   6.00 |
| Fare          |         0 |           1.0 |  32.20 |  49.69 | 0.00 |   7.91 |  14.45 |  31.0 | 512.33 |

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

- I will also remove passenger ID and name, this variable can be brought
  back later on but should not have any predictive capacity.

``` r
# Note that at this point I am only recoding or removing variables. Centering and imputation will take place later on.

train = train_raw%>%
  select(-PassengerId, -Name)%>%
  mutate(Pclass = as.factor(Pclass))

test = test_raw%>%
  select(-PassengerId, -Name)%>%
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
categorical variables based on unique categories). There are many other
forms of transformations designed to reduce skew, normalize
distributions, make features linearly separable, and more. For this
demonstration, I will not be using this advanced techniques but it is
worth acknowledging they exist.

Additionally, in this case we do have missing data in the age feature.
As part of the pipeline, we will need to specify how we want to manage
these missing entries. For the purposes of this simple demonstration, I
will impute the training average for age. Note that there are far more
advance means of managing messing data but for the sake of simplicity I
will only impute the mean.

## Building models

## Testing models
