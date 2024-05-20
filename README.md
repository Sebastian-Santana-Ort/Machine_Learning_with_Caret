# Machine_Learning_with_Caret

The purpose of this project is to demonstrate how to build a machine learning pipeline and train a predictive model in R using the [caret package](https://topepo.github.io/caret/index.html). For the purposes of this simple demonstration, I used data from the [Titanic prediction](https://www.kaggle.com/competitions/titanic/overview) competition in Kaggle.

In the end, I was able to build a series of models and compare their ability to predict accurately whether a passenger in the Titanic would survive or not. The [final model selected](https://github.com/Sebastian-Santana-Ort/Machine_Learning_with_Caret/blob/main/Caret_Demonstration.md) predicted survival with 80% accuracy.

![alt_text](https://github.com/Sebastian-Santana-Ort/Machine_Learning_with_Caret/blob/main/Caret_Demonstration_files/figure-commonmark/unnamed-chunk-3-1.png?raw=true)

In this project, you will be able to find the following:
- Markdown file named "Caret_Demonstration.md" You should **_open this file first._**
  - Here you will find a comprehensive explanation of the entire process of
    1. Loading and cleaning data,
    2. Building a machine learning pipeline,
    3. Training a series of models (using cross-validation to tune hyperparameters),
    4. Comparing models based on their accuracy, and
    5. Evaluating the performance of the final model against the testing dataset.
- Quarto markdown file named "Caret_Demonstration.qmd"
- An R project file "Machine_Learning_with_Caret.Rproj", this file will allow you set up your working environment and should be set up first if you are looking to replicate my work.
- Lastly, you will find two folders. The "Data" folder contains the original data I downloaded from Kaggle. The "Caret_Demonstration_files" contains the images made when rendering the markdown file in Quarto.
