# cse151_project
Group Project Repository for CSE 151A

https://colab.research.google.com/drive/1Ha5x6roQLp1FGVMqwziU4At1Q1Xk8KwX#scrollTo=0G1AfZZH0gtO

For our preprocessing, we will not need to impute any missing values because we found that there were no null values during our data exploration. We will normalize the 'Volume', 'Open', 'Close', 'High', and 'Low' columns because the 'Volume' data is skewed and the range of the data is a larger than the other numerical volumes. To do this, we will use min-max normalization. This will make the data easier to analyze and prepare it for machine learning models. Since 'Open time' is of type datetime64, we will extract features such as year, month, day, and hour of the day from it for additional information. Since there are no categorical variables, we will not need to do any encoding.

In Milestone 3, we decided to reduce our dataset to 1 million data points due to its increasing size and the daily addition of new data. We began preprocessing the data by applying a log transformation to the volume column due to outliers and then performed min-max scaling on all of the features. After preprocessing, we created a model using polynomial regression and generated a fitting graph to identify the optimal degree. Upon reviewing the fitting graph, we answered the required questions, determined the next model to try, and wrote a conclusion.
