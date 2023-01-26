
# Quora Check Similar Questions

## Problem Statement:  

Identify given a pair of questions are similar or not.

## Description:

Quora is a place to gain and share knowledge about anything. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask multiple questions with same meaning. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question.

## Data: 
The dataset used in this project is the Quora Question Pairs dataset from Kaggle. It contains 4,04,290 question pairs. The ground truth is the set of labels that have been supplied by human experts.

Data Source: https://www.kaggle.com/competitions/quora-question-pairs/data

## Model:
The model used in this project is XGBoost (eXtreme Gradient Boosting). It is an open-source implementation of the gradient boosting framework for decision trees.
Gradient Boosting is an iterative algorithm that tries to minimize the loss function by adding new decision trees to the model. The algorithm starts with a simple model, such as a single decision tree, and then iteratively adds new trees to the model while trying to improve the predictions. The predictions of the new trees are added to the predictions of the existing trees, and the final prediction is the sum of the predictions from all the trees.

XGBoost is a powerful and efficient machine learning algorithm that is widely used in industry and academia. It is particularly effective for large datasets and problems with many features, and it is a popular choice for machine learning competitions and real-world applications.

## Evaluation:
The model's performance is evaluated using metrics such as Log loss, confusion matrix, precision and recall.

## Solution Provided:

Model will classify a new question is similar to previous questions or not. If it is similar to previous questions, model will show previously similar question and it's answer and if it's not similar, model will add it as a new question into database.

## Tools used

Python, SQL, NLP, ML, pandas, NumPy, matplotlib, scikit learn, git, flask, docker, AWS.

## Feature Extraction Flow

![1](flowcharts/feature%20extraction.png)

## Training Flow

![2](flowcharts/model%20training.png)
