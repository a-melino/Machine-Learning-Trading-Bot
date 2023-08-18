# Module-14-Assignment - Machine Learning Trading Bot

#### by Alex Melino

#

## Overview of the Analysis

This assignment covers algorithmic trading using machine learning. More specifically, the assignment entails manipulating a dataset to create a predictive algorithm and then applying machine learning techniques to evaluate the model and predict performance. Fine tuning and applying new classifier methods is then done to acheive better results from the baseline.

The dataset used (found in the 'Resources' folder of this repo as 'emerging_markets_ohlcv.csv') consists of fictional stock or portfolio data. The assignment (found in this repo as 'machine_learning_trading_bot.ipynb') consists of reading and cleaning the data, establishing an algorithm based on short and long moving averages, splitting data into test and training sets, applying machine learning, evaluating results, and then fine tuning and applying new classifier methods tofurther improve results.

The baseline model results and the results of all attempts at improvements are stored as .png files in the 'Resources' folder of this repo.

The libraries and dependencies used in this analysis are numpy, pandas, pathlib, sklearn, matplotlib, and hvplot.

## Results
#

![Baseline Algorithm](Resources/baseline.png)

* Baseline Algorithm - SVM Classifier:
  * Short Window SMA: 4
  * Long Window SMA: 100
  * Training Period: 3 months
  * Actual Returns: ~1.39
  * Strategy Returns: ~1.51
  * Strategy Improvement vs Actual: ~8.6%

#
#
![Alternative Algorithm 1](Resources/alternative1.png)

* Alternative Algorithm 1 - SVM Classifier:
  * Short Window SMA: 4
  * Long Window SMA: 100
  * Training Period: 6 months
  * Actual Returns: ~1.56
  * Strategy Returns: ~1.83
  * Strategy Improvement vs Actual: ~17.3%

#
#
![Alternative Algorithm 2](Resources/alternative2.png)

* Alternative Algorithm 2 - SVM Classifier:
  * Short Window SMA: 4
  * Long Window SMA: 150
  * Training Period: 6 months
  * Actual Returns: ~1.50
  * Strategy Returns: ~2.10
  * Strategy Improvement vs Actual: ~40.0%

#
#
![AdaBoost Classifier](Resources/adaboost.png)

* Alternative 3 - AdaBoost Classifier:
  * Short Window SMA: 4
  * Long Window SMA: 100
  * Training Period: 3 months
  * Actual Returns: ~1.39
  * Strategy Returns: ~1.59
  * Strategy Improvement vs Actual: ~14.4%

#
#
## Summary

In order to compare the various results from each scenario above, one has to compare each strategy's outcome to the outcome of the 'Actual Returns" which is just the regular returns one would get over that same time frame. The time frames differ depending on the parameters of each algorithm so the best metric to compare them is the 'Strategy Improvement vs Actual'.

The first alternative was to adjust the size of the training set for the algorithm. It was adjusted from 3 months to 6 months, and the first alternative provided 17.3% better returns compared to the baseline of 8.6% better returns vs. actual.

The second alternative was to adjust the sizes of the SMA windows for the algorithm. Adjusting the short window in either direction seemed to only provide worse results, so it was left unchanged. However, the long window was adjusted from 100 to 150 and this seemed to provide the best results. The second alternative provided 40.0% better returns compared to the first alternative of 17.3% and the baseline of 8.6% better returns vs. actual.

The final iteration had the algorithm and training set parameters set the same as the baseline model (ie. 4 and 100 short and long SMA windows, and a 3 month training period) but this time a different classifier was used. The AdaBoost classifier was used as an alternative to the SVM classifer used previously. This yielded an improvement of 14.4% better returns compared to the baseline of 8.6% better returns vs. actual. 

The Adaboost classifer by itself did not provide better results than Alternative 1 or 2, but with fine tuning, it can be expected that this classifier should be a better match for this data set than the SVM classifier.