# Web Traffic Forecasting with ARIMA

This capstone project is a continuation of the project "[Web Traffic Forecasting](https://github.com/akhi2908/web-traffic-forecasting)" which was a completed for Udacity's Machine Learning Nanodegree Course. 

Companies such as Google and Facebook are interested in having models which can predict web traffic to certain websites potentially for better resource distribution and ad placements. This project is based on a competition floated by Google on Kaggle.com about a year ago. The objective is to explore the web traffic project and create a time-series model which can predict web traffic for hundreds of wiki articles. 

Forecasting future events is a valuable many institutions such as stock markets, online sellers and advertisers. Forecasting web traffic is one of the most challenging problems the data has not just strong temporal effects but also dependencies on current events which are difficult to capture. 

The goal of the project is to predict the web traffic on the Wikipedia based on previous views for hundreds of wiki articles. Exploratory analysis will be performed on the dataset to find relevant information. A dataset will also be split and merged to find total daily views for each language. There will be two types of predictions made:
1.	Prediction for daily views for every page; and, 
2.	Prediction for daily views by language.
For forecasting, the dataset will be divided into a training set and testing set and validation set. The final prediction will be made for about a month (50 days). The quality of the predictions made by the model will be evaluated using Root Mean Square Error (RMSE).


## Getting Started

### Files needed

* Jupyter Notebook: WebTrafficExplorationARIMA.ipynb

* input csv files: https://drive.google.com/drive/folders/13SoRSaZ4fcdsXE0eI0BukKKxdinJUdJG?usp=sharing

(Additional files can be found here: [Web Traffic Forecasting](https://github.com/akhi2908/web-traffic-forecasting))

### Prerequisites

* Python 3

* Jupyter notebook

* The following libraries need to be imported 

```
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from math import sqrt
```

## Authors

* **Akhilesh Jain** - [akhi2908](https://github.com/akhi2908)

## References

* Understanding basic concepts behind statistical time series analysis: https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm

* Full notes on statistical forecasting: http://people.duke.edu/~rnau/411home.htm

* When to use which forecasting model: http://people.duke.edu/~rnau/whatuse.htm 

* Good python tutorial on TSA
https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-9-time-series-analysis-in-python-a270cb05e0b3
http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/

* Statsmodel library: https://www.statsmodels.org/stable/tsa.html

* Exponential Smoothing: https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html
