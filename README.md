# Predictive Modeling for Crop Prices in Online Markets with Sparse Historical Data

This is a research project designed by Professor Divya Singhvi at NYU Stern School of Business, with my implementation and report analysis. All datasets are provided by Professor Singhvi and are confidential for view only.

## Introduction

We aim to predict future prices for 151 crops across 212 online markets, with a sparse historical transaction data from 2014 to 2019. We utilized three datasets: Transaction Data, Mandi Characteristics, and Crop Characteristics.

Sparse historical data poses a challenge as price information may not be available for every crop in every market due to farmers' irregular visits and seasonal crop availability. To create precise one-day ahead price predictions for each crop in every market, we first conduct exploratory data analysis and employ missing-value imputation techniques. Then, we select four predictive machine learning models - CART, [PTF](https://www.cs.cmu.edu/~jgc/publication/PublicationPDF/Temporal_Collaborative_Filtering_With_Bayesian_Probabilidtic_Tensor_Factorization.pdf) (implementation spearheaded in this project), Random Forest, and Convolutional Neural Networks, and evaluate their performance.

## I. Explore Data

We first addressed key questions to guide our selection of the prediction model:

1. [ ] Whether geographically proximate markets exhibit correlated crop prices.
2. [ ] If certain markets demonstrate higher price variability compared to others.
3. [ ] If there exists a dominant market influencing other markets.
4. [ ] Correlations among crops to market correlations.

---
### 1. Whether geographically proximate markets exhibit correlated crop prices
We grouped markets by district IDs. Then we identified the most traded crop in each district based on transaction data from 2014 to 2019. Next, we selected the two markets trading this crop most frequently in this district. Using intersected transaction dates, we calculated the correlation coefficient between these two markets' prices for the chosen crop over time. This process was implemented in `PriceCorrOverCloseMarkets()` in `explore_data.py`. Output as follows:

```
Correlation between mandi 3 and mandi 9 over crop 17 is 0.80.
Correlation between mandi 4 and mandi 22 over crop 6 is 0.65.
Correlation between mandi 63 and mandi 187 over crop 4 is 0.59.
Correlation between mandi 16 and mandi 117 over crop 6 is 0.47.
Correlation between mandi 122 and mandi 126 over crop 4 is 0.52.
Correlation between mandi 147 and mandi 25 over crop 4 is 0.54.
Correlation between mandi 33 and mandi 41 over crop 2 is 0.83.
Correlation between mandi 150 and mandi 152 over crop 21 is 0.64.
The average price correlation over 8 pairs of close markets is 0.63.
```

Our analysis revealed that out of 8 pairs of markets, all but one exhibited correlations exceeding 0.5, with two pairs exceeding 0.8, indicating a notably strong positive correlation. On average, correlations hovered around 0.63. We concluded that geographically proximate markets tend to demonstrate price co-movement for their crops.

### 2. If certain markets demonstrate higher price variability compared to others
Here we conducted an exploration on a market-by-market basis. For each market, we identified the most traded crop and calculated the variance of its daily weighted price. This procedure was implemented in `MarketVariability ()` in `explore_data.py`. The output is:

```
Of all markets, the minimum price variance for a single market is 0.45, the 25th quantile price variance is 13.71, median price variance is 30.82, the 75th quantile price variance is 97.77, the maximum price variance is 185.79.
```

We observed a 75th quantile variance of 97.77, a maximum variance of 185.79. Therefore, it is reasonable to deduce that some markets are much more variable than the others.

### 3. If there exists a dominant market influencing other markets
We examined the impact of a particular market on all other markets. For each market, we identified the crop traded most frequently and calculated its weighted price across all other markets. Subsequently, we conducted a correlation analysis between the price of this market's crop (shifted one day ahead) and the weighted crop price of all other markets, determining if this market exerted influence on others. Analysis was conducted using intersected transaction dates and implemented in `InfluentialMarket ()` in `explore_data.py`.

The maximum correlation achieved through this method was 0.51, attributed to market ID 22. However, upon examining the correlation table between market 22 and other market for crop 6's price, we observed that only data from the years 2017 and 2018 were utilized, indicating a limited temporal correlation strength. We also analyzed a correlation table for market 3 and other markets regarding crop 2's price:

```
The price correlation table over crop 2 between market 3 and other markets is
          date  weighted_price_x  weighted_price_y
0   2015-12-13               5.0          5.838832
1   2015-12-18               7.0          7.441011
2   2015-12-20               7.0         10.615118
3   2015-12-26               9.0         10.000000
4   2015-12-29              11.0         10.773043
..         ...               ...               ...
577 2018-07-21               9.0          7.965580
578 2018-07-22               9.0          7.442310
579 2018-07-23               7.0          8.912302
580 2018-07-24               5.0         11.300996
581 2018-07-25              15.0          9.652697

[582 rows x 3 columns]
Correlation equals 0.48.
```

This correlation demonstrated robustness over time, spanning from 2015 to 2018 across 582 days, with a value just slightly below 0.5. However, we noted that the price for this crop remained relatively stable around 8.0. Given these observations, we inferred that this crop might be a daily necessity, which is traded frequently with a consistently low price. Consequently, concluding that market 3 is an influential market would be inappropriate, as all markets may exhibit similar price fluctuations for such an essential crop.

### 4. Correlations among crops to market correlations
We identified the top 5 traded crops based on trading days and conducted correlations for every pair of them if their intersected trading days exceeded 400. This task was completed in `CropCorrelation ()` in `explore_data.py`.

Among the outputed 6 pairs of crops, 1 pair has no correlation, 3 pairs have slightly positive price correlation, 1 pair has slightly negative price correlation, and 1 pair has a strong positve correlation (crop 4 and crop 9) with a Pearson correlation of 0.63.

## II. Predictive Models and Evaluation

Now we aim to build machine learning models to predict one-day behind crop price. Following our data exploration insights, we identified that crop prices correlate strongly with categorical traits like market location and crop type. Consequently, we selected two models, Classification and Regression Trees (CART) and Random Forest (RF), for predictions heavily based on categories. 

Another model we chose is Temporal Collaborative Filtering using Probabilistic Tensor Factorization (PTF), built based on Equation 3.9 in the paper "[Temporal Collaborative Filtering with Bayesian Probabilistic Tensor Factorization](https://www.cs.cmu.edu/~jgc/publication/PublicationPDF/Temporal_Collaborative_Filtering_With_Bayesian_Probabilidtic_Tensor_Factorization.pdf)". The reason for choosing PTF is that it is suitable for time-related sparse data. Since markets are commonly sensitive to their previous prices, another model we chose is simply the past mean price of the crops if available; if not, we filled it with the average price of all crops. The last model we chose is the Multi-Layer Perceptron Neural Network (CNN).

### Train-test Setup and Model Evaluation
We divided the transaction data from 2014 to 2018 for training and data from 2019 for testing. Treating dates as "month-day" and disregard year, we utilized the weighted price of a crop for the same month, day, and market, regardless of the year. For CART, RF, and CNN, we selected categories including "date (no year)", "crop ID", "mandi ID", "District ID", "Latitude", "Longitude", and "Type" (with 1 indicate an open market). In addition, for PTF, we split the training sets into eight subsets based on "date (no year)", "mandi ID", and "crop ID" to enhance model accuracy; these subsets were used to construct market features, crop features, and time features as stated by Eq. 3.9 in the [paper](https://www.cs.cmu.edu/~jgc/publication/PublicationPDF/Temporal_Collaborative_Filtering_With_Bayesian_Probabilidtic_Tensor_Factorization.pdf). 

Consequently, the testing sets were also split into eight subsets using the same criteria, and the mean absolute error (MAE) of five models were assessed within each subset.

Finally, we evaluated the average MAE across the entire testing set. The whole process was implemented in `predict.py` with the following results.

![result](https://github.com/yaodan-zhang/ptf-algo/blob/main/result.png)

We found that, among the seven rounds where the eighth round's testing dataset is empty due to its lackness, PTF's performance remains comparable to CART (RF) and Mean, and is most of the time better than the performance of CNN. One exception is in round 3, where the instability of its performance might come from a bad approximation of the optimal log likelihood in the model, because this likelihood function is non-convex and therefore the minimizer only loops to a local but not global minimum. However, one potential gain using collaborative filtering (PTF) is that, the trained feature matrices can be applied to similar scenarios when historical data of a market over a crop is completely unavailable. In this case, past mean is no longer available, and PTF supplies the need. PTF can also capture possible trends emerging over time with its time feature, but in our setting Mean and CART (RF) seems to be good enough.

## III. Discussion

Time division is another crutial factor that matters in model performance, as here we chose year 2014 to 2018's data to be the training set and year 2019 to be the testing set. This 5:1 train-test ratio is good enough for a typical machine learning technique, but what if the traning data becomes more and more sparse? Future evaluations can focus on the impact of sparsity threshold on model performance and potentially demonstrate the superiority of the PTF technique.

## Reference

Xiong, Liang, et al. "Temporal collaborative filtering with bayesian probabilistic tensor factorization."  *Proceedings of the 2010 SIAM international conference on data mining* . Society for Industrial and Applied Mathematics, 2010.
