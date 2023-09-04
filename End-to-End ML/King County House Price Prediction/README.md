# King-county-House-Price-Prediction-ML
King County House Price Prediction using 5 Different Machine Learning Model based on Regression 

## Context
King county house price is a dataset which contain information about house price in King County area and information about the property specification.

This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015. The attributes are deﬁned as follows (taken from the UCI Machine Learning Repository):

    id - Unique ID for each home sold
    date - Date of the home sale
    price - Price of each home sold
    bedrooms - Number of bedrooms
    bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower
    sqft_living - Square footage of the apartments interior living space
    sqft_lot - Square footage of the land space
    floors - Number of floors
    waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not
    view - An index from 0 to 4 of how good the view of the property was
    condition - An index from 1 to 5 on the condition of the apartment,
    grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.
    sqft_above - The square footage of the interior housing space that is above ground level
    sqft_basement - The square footage of the interior housing space that is below ground level
    yr_built - The year the house was initially built
    yr_renovated - The year of the house’s last renovation
    zipcode - What zipcode area the house is in
    lat - Lattitude
    long - Longitude
    sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors
    sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors

## Problem Statement
As a property agent, it is crucial to know the best value of a house so we could get highest profit from selling house and lowest investment when buying house.

We could infer the main problems that we could solve with predicting King County house price with machine learning model is to know the house price estimation based on the house specification so we have a better standing if we want appraise house and get the best price.

## Goals
- Create a machine learning model using Linear Regression to predict King County House Price based on house specification
- Improve machine learning model using Polynomial Features
- Improve machine learning model using Ridge/Lasso/Elasticnet
- Create evalution matrix to determine the best machine learning model to predict King County House Price

## Summary
The best Machine Learning Model in this project to use to predict King County House Price is Ridge with Hyperparameter Tuning (alpha = 0.0001) with R2 score 0.80 and MAE score 103505
