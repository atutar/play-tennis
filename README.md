# Tennis Decision Tree Classifier
This project implements a decision tree classifier to predict whether to play tennis based on weather conditions.

## Project Overview
The program analyzes a dataset containing weather conditions (outlook, temperature, humidity, wind) and historical decisions about whether to play tennis under those conditions. Using this data, it builds a decision tree model that can predict whether tennis should be played given new weather inputs.

## Features
* Data preprocessing using pandas
* Decision tree classification using scikit-learn
* Model evaluation with accuracy and classification metrics
* Visual representation of the decision tree using Graphviz

## Dataset Description
The dataset consists of the following attributes:

* outlook: Sunny, Overcast, Rain
* temperature: Hot, Mild, Cool
* humidity: High, Normal
* wind: Weak, Strong
* play: Yes, No (target variable)

## Usage
### Install the required dependencies:
* pip install pandas scikit-learn graphviz pydotplus

### Make sure Graphviz is installed on your system:
* Windows: Download from the official Graphviz website
* macOS: brew install graphviz
* Linux: sudo apt-get install graphviz

### Add Graphviz to your system PATH  

### Run the Jupyter notebook or the Python script:
* jupyter notebook play_tennis_or_not.ipynb

## Datasets used
* https://www.kaggle.com/datasets/fredericobreno/play-tennis?select=play_tennis.csv
