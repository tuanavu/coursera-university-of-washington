### Regression Week 1: Simple Linear Regression Assignment

Predicting House Prices (One feature)

In this notebook we will use data on house sales in King County, where Seattle is located, to predict house prices using simple (one feature) linear regression. You will:

- Use SArray and SFrame functions to compute important summary statistics
- Write a function to compute the Simple Linear Regression weights using the closed form solution
- Write a function to make predictions of the output given the input feature
- Turn the regression around to predict the input/feature given the output
- Compare two different models for predicting house prices

### If you are doing the assignment with IPython Notebook

An IPython Notebook has been provided below to you for this assignment. This notebook contains the instructions, quiz questions and partially-completed code for you to use as well as some cells to test your code.

### What you need to download

#### If you are using GraphLab Create:

- Download the King County House Sales data In SFrame format: [kc_house_data.gl.zip](https://eventing.coursera.org/api/redirectStrict/xcnzdgbObEMu4g1yr2xX9oe4aC-giHGZa8IcmV8fRdfcOv1u_U-BaOm0riqQ6AJ_X2YEAh5jXDC5IA048c5uwQ.q51cq8rDCoVh6Qlrc8Uo7A.WJd7f3YXcv5wixsfUcmpo__qlQ708ddW0w00Mr2pQV5sY0eHpdsK0jNPy1AXF9e_OOi9iDtv6YZ8LO6kQ-qUajRWB6ia3DdOIBPKIfplbfXlwvfQmF0VUoMyWlZw2vQzHZ-DqAqc6WyxbKVO0YQOiKlgGzwLCtudZH7WtmuR11Dz_OqGF_hyQCNHaCgW8K4mxk7YhKqKFXWtfO10cePQBdZij9t7ZtXBXC4ReakfJugBr1AGjPSQVCxe8xgDQ03aAZQLhPKRMeux0RIn14EWR25zPi8ivzaPUP8SE-_zlcANKYEIVpJtQwZWSTU3GTjVsaNZZ85N-GftiJfQQ4AZtiNP7y3OxPjIW2cIVxfGSQC-cWNuJE4so8MdtzhA7VdI_rEva1wkxdjZ4qYHBGRAFFqeIown_B1eFbKUJlMRmlePE4-tDzyioLyUULycacu2)
- Download the companion IPython Notebook: [week-1-simple-regression-quiz-blank.ipynb](https://eventing.coursera.org/api/redirectStrict/c_mg8mJyfRYhu3RKd4s-bwEXoJGioq_XUKhTkYvyNy6UtNg1LBViIE5VahOcV_C82eHgww0KAPyGtocRmzPvJQ.llvMdy-JGD4u8HxeCo2RpQ.4fdHv8Dopng9k6Aol4oJ2CAMZNDXG9Gx5H3kW8YUbZa_O572yqUKGuTsWNG98vfhfBoGXf6vVXpF1fJu8XPtbf-khk-FPP7gAV26gAepK_bYqyv4FjJP9TvL2UdIxz9TvIGPfzARejGxEZH5pOJClpnd5RC5W9FiVqNXpI1C5E8wnkHETfIKCHcrsUv1nStauipItwSmIJV_FJcm5rqF8ZsKD6dWR36rxePqa-3rtsmtG-b7KgcqCEpmjcwMpA2GRZdw_3PR6zAKgXEmsoAVl1WGOxXdCMDOKJTdR8hPFtvOgeiY_czl8tgi-VfmwlKTuxtM-vzjPHXMs0xQZyDRSs-T04eQikkkQH6Y8X-YjQEtvgMieXpSiwGd_z-v96zS5FTlbck0zw1uVaYgOciIqJGLoThwTjJ01BefBxThREvdCOvgjSjfGXX8iSQcetefirshvy8SUNMsFE17r_7umJh5_GD6lPhSYECCq-FElKk1zKYgdt5IOYx_sDmUyZ_3)
- Save both of these files in the same directory (where you are calling IPython notebook from) and unzip the data file.

#### If you are not using GraphLab Create:

- Download the King County House Sales data csv file: [kc_house_data.csv](https://eventing.coursera.org/api/redirectStrict/ETG2i1rAouT3v7XZfL98lW7dkshyXso2XIHCUfFrWURncLwZT9WaUN95RHDQ0rMmFPp4SuNyVTvgdQda-0PeOQ.IM6r16X2XAqPbWgNBpKKsw.T_mDUCfRoYMiE3ZATOhK0NVMKJVWsxCDnAMTF9f4Zg41k50XSVG1yjyj-b1T-rbynWCm7kHP2u65I_ui8MziQclhqH0zZe3y_BxhtsTABWcgV6GYYduw_w6E9nU3SADyjwQ6csrWrorMFEr4WEl8AzRq60mQtHtNtids5zWxuMCcw4FeWJ38rmH7nUi-JQu2Awe_123Siv5YnpBQQEjaBIJWpnW7SMMmnEOsE2zWAweS2x2H9hWkbIROhVf6eyLz59JdluEfN_3v9Ffl_MJ5kqMGqdmwErbTnFXC1V_npDSaHl0iBLXk_vTy85xHqnygVQzMsNX1HSCBx5OtmhMkF2ge0HHktDH3EAV-j0MNAI6sH6FnCzAlgQfD7pRfHw6DtlDCUwK2YD4MBuT9F8Gz18DLacFiOQG1X-fO2OyJYHcEKgy3zUNcBN4j-IDRRlak)
- Download the King County House Sales training data csv file: [kc_house_train_data.csv](https://eventing.coursera.org/api/redirectStrict/AzdhSuE3cmgsOUm7tIsIkPCVUbkEa8Z9xR0PWbTww199RHtkzE9cxS9cX-kgAG3P3Bc9ssD01vddyleKIDQVgw.5Z-JU2bCalfpNbl6lzjJ4A.w016ZB7z-AYwRSK4AZdPTKewoDb2fAb5AQwlKhpF-wYWuUq1fDh0C1A24kU3Wv79RWZCB6dTshgjXdKcnXAQt8NBn5c37-OiRjovxjjHuUsLBOYyWFf8iy4wRJSauqSPf-SOX7k85ePjzQ7HZX8DmY7KqrwRsOj4WG5NUcOxLx0umCFkLqKEm6byZ7p4hh0BbodQO_8XLayUw_Dh12uR1KRG916uFh7_IuApr_0iOwDGchrH9Cvp3eIpaYG3R9_LUQORdWRE0cDvSeFBEy9AuQwAt4L63KgudbrVAlUSte8vZJ9TGMER8FsINKbGO8tX8e-gourbIyOdrPI9HRDBGQlfgMbDFZT-HsJUJ8Vw0S1RnlvB3F46m2Ye6CsZd5Fb-uTGe88TFx2jwOB8My83-t6kLZJCIuZyXji-oObU74IchmxNgLnq43llkjCRjibS4CCfl_u-8FzRNCgLPeK6SA)
- Download the King County House Sales testing data csv file: [kc_house_test_data.csv](https://eventing.coursera.org/api/redirectStrict/-2-uL_cYaYOYmuWzWGWpbcHBreNOXyumEFCxCCLtZjj2-Pqi0F9_yXSCeDA6yBd-1mi28GuZ0PYacmkaYvzZ_w.6LQp3XPPiLJdmoWtVzsJ2w.nwM30Y_2KXR2rr2qutRXDuoPCNohY4QtNhr48dVFnhOTWVKCY9GVEpvbjjVX3TaviH4uk1ETyNeyYCVGqa6sGp-ntuluoduJLasXEj7DrbNkpmyToczTvF0TIG1hp3WWmXcEEyQQ_MLzaxP9KBbBWqzaAOu_FCjQdV69xVSjr0serK9yvkn0VYnIxDI4fQ8KpR88MWiOAa8HnY-SVg4P8XVDEK-kXvH_vJyFVpS_Dt_iVXQ9JOagsYlB_E6Fh6ZDCkBHDoeM3FDPlteefpCHu1DeFDyngM9Dfmw0K622hpAIbTsJRAAv-ZJcIQZBzpc2NjMCbo3X_5h2O-9fDQIn6mqNJWjgjANwXh7hLipFWM3nML-gWik8Pl6NjxxLiAPx587M1IGsCD011srf1hPm0mNu6nY9Lf_1HsWBTO50hnmxtzI9ovNfZDDftHk1R2Jj)

### Useful resources

You may need to install the software tools or use the free Amazon EC2 machine. Instructions for both options are provided in the reading for Module 1.

### If instead you are using other tools to do your homework

You are welcome, however, to write your own code and use any other libraries, like Pandas or R, to help you in the process. If you would like to take this path, follow the instructions below.

**1.** If you are using SFrame, import graphlab and load in the house data, otherwise you can also download the csv. (Note that we will be using the training and testing csv files provided). e.g in python with SFrames:

```python
sales = graphlab.SFrame('kc_house_data.gl/')
```

**2.** Split data into 80% training and 20% test data. Using SFrame, use this command to set the same seed for everyone. e.g. in python with SFrames:

```python
train_data,test_data = sales.random_split(.8,seed=0)
```

For those students not using graphlab please download the training and testing data csv files.

From now on we will train the models using train_data. It will be important that we use the same split here to ensure the results are the same.

**3.** Write a generic function that accepts a column of data (e.g, an SArray) ‘input_feature’ and another column ‘output’ and returns the Simple Linear Regression parameters ‘intercept’ and ‘slope’. Use the closed form solution from lecture to calculate the slope and intercept. e.g. in python:

```python
def simple_linear_regression(input_feature, output):
    [your code here]
return(intercept, slope)
```

**4.** Use your function to calculate the estimated slope and intercept on the training data to predict ‘price’ given ‘sqft_living’. e.g. in python with SFrames using:

```python
input_feature = train_data[‘sqft_living’]
output = train_data[‘price’]
```

save the value of the slope and intercept for later (you might want to call them e.g. squarfeet_slope, and squarefeet_intercept)

**5.** Write a function that accepts a column of data ‘input_feature’, the ‘slope’, and the ‘intercept’ you learned, and returns an a column of predictions ‘predicted_output’ for each entry in the input column. e.g. in python:

```python
def get_regression_predictions(input_feature, intercept, slope)
    [your code here]
return(predicted_output)
```

**6. Quiz Question: Using your Slope and Intercept from (4), What is the predicted price for a house with 2650 sqft?**

**7.** Write a function that accepts column of data: ‘input_feature’, and ‘output’ and the regression parameters ‘slope’ and ‘intercept’ and outputs the Residual Sum of Squares (RSS). e.g. in python:

```python
def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    [your code here]
return(RSS)
```

Recall that the RSS is the sum of the squares of the prediction errors (difference between output and prediction).

**8. Quiz Question: According to this function and the slope and intercept from (4) What is the RSS for the simple linear regression using squarefeet to predict prices on TRAINING data?**

**9.** Note that although we estimated the regression slope and intercept in order to predict the output from the input, since this is a simple linear relationship with only two variables we can invert the linear function to estimate the input given the output!

Write a function that accept a column of data:‘output’ and the regression parameters ‘slope’ and ‘intercept’ and outputs the column of data: ‘estimated_input’. Do this by solving the linear function output = intercept + slope*input for the ‘input’ variable (i.e. ‘input’ should be on one side of the equals sign by itself). e.g. in python:

```python
def inverse_regression_predictions(output, intercept, slope):
    [your code here]
return(estimated_input)
```

**10. Quiz Question: According to this function and the regression slope and intercept from (3) what is the estimated square-feet for a house costing $800,000?**

**11.** Instead of using ‘sqft_living’ to estimate prices we could use ‘bedrooms’ (a count of the number of bedrooms in the house) to estimate prices. Using your function from (3) calculate the Simple Linear Regression slope and intercept for estimating price based on bedrooms. Save this slope and intercept for later (you might want to call them e.g. bedroom_slope, bedroom_intercept).

**12.** Now that we have 2 different models compute the RSS from BOTH models on TEST data.

**13. Quiz Question: Which model (square feet or bedrooms) has lowest RSS on TEST data? Think about why this might be the case.**