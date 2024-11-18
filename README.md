# cse151_project
Group Project Repository for CSE 151A

## [Link to dataset](https://www.google.com/url?q=https://www.kaggle.com/datasets/imranbukhari/comprehensive-ethusd-1m-data/data&sa=D&source=docs&ust=1730502587695059&usg=AOvVaw3kMqZe-yQhr2LT-L_PQyeM)

For our preprocessing, we will not need to impute any missing values because we found that there were no null values during our data exploration. The data is originally represented in 1 minute intervals with `Open`, `High`, `Low`, and `Close` prices, but there is very minimal change to be found within such a short interval. Thus, we condense our data into 10 minute intervals by manually calculating the corresponding price points. In addition, we filter our data to only contain rows from the year 2023 to limit the size of our dataset to fit github constraints. We then log transform the 'Volume' column because it is skewed and the scale of the data is larger than the other numerical volumes. This will make the data easier to analyze and prepare it for machine learning models. Since there are no categorical variables, we will not need to do any encoding.

In Milestone 3, we decided to save a snapshot and reduce our dataset to 300 thousand data points due to its constant increasing size by the daily addition of new data. After preprocessing, we created a model using linear regression and plotted our coefficients and intercepts to visualize our model's performance. Upon reviewing the graph and evaluation metrics, we answered the required questions, determined next steps to try with our model, and wrote a conclusion.
