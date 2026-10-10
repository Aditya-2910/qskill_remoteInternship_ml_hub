# House price prediction

## The dataset
The regression model is trained on the California Housing dataset, built from the 1990 US census. Each row describes one census block group, which is a small area of roughly 600 to 3,000 people, not a single house. The target is the median house value in the block group, in units of 100,000 US dollars. Values are capped at 5.0, so nothing above 500,000 dollars appears in the data.

## The eight features
MedInc is the median household income in the block group, in units of 10,000 dollars, so 3.5 means 35,000 dollars. HouseAge is the median age of houses in years. AveRooms is the average number of rooms per household and AveBedrms the average number of bedrooms per household. Population is the number of people living in the block group. AveOccup is the average number of people per household. Latitude and Longitude give the location of the block group inside California.

## What drives price
Median income is by far the strongest predictor: richer neighbourhoods have more expensive houses. Location is next, because coastal areas, especially around the San Francisco Bay Area and Los Angeles, are much more expensive than the inland valleys and desert regions. Average rooms adds some signal. Very high average room counts sometimes indicate holiday-home areas with few permanent residents rather than large family houses.

## Models and metrics
Linear Regression is the classic baseline. It assumes each feature has a straight-line effect on price. Gradient boosting can capture non-linear effects such as the coastal premium and usually gives a lower error. Mean Squared Error is the average squared difference between predicted and actual prices, and RMSE is its square root, which is in the same unit as the target. R squared is the fraction of variance in prices the model explains, where 1.0 is perfect and 0.0 is no better than predicting the mean.

## Limitations
The data is from 1990 and the predictions are for block-group medians, so they are not appraisals of individual houses and do not reflect todays market. Predictions for inputs far from the training data, for example a location outside California, should not be trusted. Prices are capped at 500,000 dollars, so the model cannot predict above that level.
