# Clickthrough Rate Prediction

This is the final project for class Machine Learning at Scale. In this project we attempted the [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) in 2014 which is to train a model with 10GB of web traffic data to predict click-through rate (CTR, percent of ads clicked). 

Team: Anu Yadav, Connor Stern, Erik Hou, Noah Pflaum

## Description
Inspired by the [winning team](https://www.kaggle.com/c/criteo-display-ad-challenge/discussion/10555)'s approach, we explained the concept of field-aware fectorization machine (FFM) and sought to build our own homegrown FFM model with PySpark's RDD API. The primary goal is to beat a baseline model defined by measuring the overall click-through rate. A secondary goal is to match the model performance of competition-winning models.

For the details of our approach and lesson learned, please refer to our [project notebook](https://github.com/erikhou45/ctr-prediction/blob/main/final_project.ipynb) 
