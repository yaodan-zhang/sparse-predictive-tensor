# Project: Predictive Modeling for Crop Prices in Online Indian Markets with Sparse Historical Data

Note: This GitHub repository is a research project designed by Professor Divya Singhvi from NYU Stern School of Business, with author's implementation.

## Introduction

This project aims to predict future prices for approximately 150 crops across 200 online Indian markets, using a sparse historical transaction data spanning from 2014 to 2019. We utilized three datasets: Transaction Data, containing daily transaction data including crop and mandi IDs, seller and buyer IDs, crop quantity, and crop traded price; Mandi Characteristics, providing mandi details such as name, district, state, location, and market type (open market or not); and Crop Characteristics, mapping crop names to IDs. Here's a summary of each dataset:

![dataset_first_part](https://github.com/yaodan-zhang/ptf-algo/blob/main/data1.png)

![dataset_second_part](https://github.com/yaodan-zhang/ptf-algo/blob/main/data2.png)

Sparse transaction data poses a challenge as price information may not be available for every crop in every mandi daily due to farmers' irregular visits and seasonal crop availability. To create precise one-day ahead price predictions for each crop in every market, we first conduct exploratory data analysis and employ missing-value imputation techniques. We then select predictive models based on four machine learning techniques (the main one called [PTF](https://www.cs.cmu.edu/~jgc/publication/PublicationPDF/Temporal_Collaborative_Filtering_With_Bayesian_Probabilidtic_Tensor_Factorization.pdf)'s development is spearheaded in this project) and evaluate their performance. Throughout the project, we employ a train-test validation method to assess model accuracy.

## Part I. Exploratory Data Analysis

In the first part, we addressed key questions to guide our selection of the prediction model:

1. [ ] We investigated whether geographically proximate markets exhibit correlated crop prices.
2. [ ] We examined if certain markets demonstrate higher price variability compared to others.
3. [ ] We identified if there exists a dominant market influencing other markets.
4. [ ] We explored correlations among crops akin to market correlations.

---

For our first question, we grouped markets by district using district IDs. Then, we identified the most traded crop in each district based on transaction data from 2014 to 2019. Next, we selected the two markets trading this crop most frequently in this district. Using intersected transaction dates, we calculated the correlation coefficient between these two markets' prices for the chosen crop over time. This process was implemented in `PriceCorrOverCloseMarkets()` in `part1.py`. With output as follows:

```
Correlation between mandi 3 and mandi 9 over crop 17 is 0.8082058671768368.
Correlation between mandi 4 and mandi 22 over crop 6 is 0.6588707894950585.
Correlation between mandi 63 and mandi 187 over crop 4 is 0.5933894315791544.
Correlation between mandi 16 and mandi 117 over crop 6 is 0.47967122154626196.
Correlation between mandi 122 and mandi 126 over crop 4 is 0.5288967137844734.
Correlation between mandi 147 and mandi 25 over crop 4 is 0.5426776295567428.
Correlation between mandi 33 and mandi 41 over crop 2 is 0.8322216322594749.
Correlation between mandi 150 and mandi 152 over crop 21 is 0.6493325813182255.
The average price correlation over 8 pairs of close markets is 0.6366582333395285.
```

Our analysis revealed that out of the eight pairs of markets where transaction data allowed correlation calculation, all but one exhibited correlations exceeding 0.5, with two pairs exceeding 0.8, indicating a notably strong positive correlation. On average, correlations hovered around 0.63. Thus, we concluded that geographically proximate markets tend to demonstrate price correlation for their respective crops.

For our second question, we conducted an exploration on a mandi-by-mandi basis. Specifically, for each mandi, we identified the most traded crop and calculated the variance of its daily weighted price. This procedure was implemented in `MarketVariability ()` in `part1.py`. The output is:

```
Of all markets, the minimum price variance for a single market is 0.4510640032548529, the 25th quantile price variance is 13.712867783124935, median price variance is 30.823880561057084, the 75th quantile price variance is 97.777898403187, the maximum price variance is 185.79842023464707.
```

We observed a 75th quantile variance of 97.77, while a maximum variance of 185.79. Therefore, it is reasonable to deduce that, some markets are much more variable than the others.

For our third question, we examined the impact of a particular market on all other markets. For each market, we identified the crop traded most frequently and calculated its weighted price across all other markets. Subsequently, we conducted a correlation analysis between the price of this market's crop (shifted one day ahead) and the weighted crop price of all other markets, determining if this market exerted influence on others. Analysis was conducted using intersected transaction dates and implemented in `InfluentialMarket ()` in `part1.py`.

The maximum correlation achieved through this method was 0.51, attributed to mandi 22. However, upon examining the correlation table between mandi 22 and other mandis for crop 6's price, we observed that only data from the years 2017 and 2018 were utilized, indicating a limited temporal correlation strength. We also analyzed a correlation table for mandi 3 and other mandis regarding crop 2's price:

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
Correlation equals 0.48521514609447636.
```

This correlation demonstrated robustness over time, spanning from 2015 to 2018 across 582 days, with a value just slightly below 0.5. However, we noted that the price for this crop remained relatively stable around 8.0. Given these observations, we inferred that this crop might be a daily cooking necessity, which is traded frequently with a consistently low price. Consequently, concluding that mandi 3 is an influential market would be inappropriate, as all markets may exhibit similar price fluctuations for such an essential crop.

For our final question, we identified the top 5 traded crops based on trading days and conducted correlations for every pair of them if their intersected trading days exceeded 400. This task was completed in `CropCorrelation ()` in `part1.py`.

Among the outputed 6 pairs of crops, 1 pair has no correlation, 3 pairs have slightly positive price correlation, 1 pair has slightly negative price correlation, and 1 pair has a strong positve correlation (crop 4 and crop 9) with a Pearson correlation of 0.63.

## Part II. Prediction Models Selection and Evaluation

In the second part, we aim to construct imputation models to address the missing values mentioned earlier, facilitating price prediction. Following our data exploration insights, we identified that crop prices correlate strongly with categorical traits like mandi location and crop type. Consequently, we selected two models, Classification and Regression Trees (CART) and Random Forest regressor (RF), for category-based predictions. Another model we chose is Temporal Collaborative Filtering using Probabilistic Tensor Factorization (PTF), which was built on Equation 3.9 of the paper "[Temporal Collaborative Filtering with Bayesian Probabilistic Tensor Factorization](https://www.cs.cmu.edu/~jgc/publication/PublicationPDF/Temporal_Collaborative_Filtering_With_Bayesian_Probabilidtic_Tensor_Factorization.pdf)". The reason for choosing PTF is that it deals with time-related sparse data, which perfectly suits our purpose. Since markets are commonly sensitive to their previous prices, another model we chose is simply the past mean price of the crops if available; if not, we filled it with the average price of all crops. The last model we chose is the Multi-Layer Perceptron regressor, one of the fundamental convolutional neural networks (CNN).

In our train-test validation setup, we divided the transaction data from 2014 to 2018 for training and data from 2019 for testing. Treating dates as "month-day" for consistency across years, we utilized the weighted price of a crop for the same month, day, and mandi, regardless of the year. For CART, RF, and CNN, we selected categories including "date no year", "crop ID", "mandi ID", "District ID", "Latitude", "Longitude", and "Type" (1 for open market, 0 for null). Additionally, for PTF, we split the training sets into eight subsets based on "date no year", "mandi ID", and "crop ID" to enhance model accuracy; these subsets were used to construct mandi features, crop features, and time features as stated by Eq. 3.9 in the [paper](https://www.cs.cmu.edu/~jgc/publication/PublicationPDF/Temporal_Collaborative_Filtering_With_Bayesian_Probabilidtic_Tensor_Factorization.pdf). Consequently, the testing sets were also split into eight subsets using the same criteria, and the MAE of all five models were assessed within each subset. Finally, we evaluated the average MAE across the entire testing set. The whole process was implemented in `part2.py` with the following results.

![result](https://github.com/yaodan-zhang/ptf-algo/blob/main/result.png)

We found that among the seven rounds where the eighth round has an empty testing dataset, PTF's performance remains comparable to the tree and the mean techniques, and is most of the time better than the performance of CNN. One exception is in round 3, where the instability of its performance might come from a bad approximation of the optimal log likelihood in the model, because this likelihood function is non-convex and therefore ”SLSQP” in the minimizer function only loops to a local but not global minimum. But one potential gain using collaborative filtering is that, the user and item features can be applied to similar scenarios when historical data of the users over an item is completely unavailable. In this case, past mean is no longer available. Apart from that, PTF can capture the possible trends with time using its time layer/feature, but in our setting mean and trees seemed to be good enough for this aspect.

## Part III. Discussion

Time division is another crutial factor that matters in model performance, as here we chose year 2014 to 2018's data to be the training set and year 2019 to be the testing set. This 5:1 train-test ratio is good enough for a typical machine learning technique, but what if the traning data becomes more and more sparse? Future evaluations can focus on the impact of sparsity threshold on model performance and potentially demonstrate the superiority of the PTF technique.

## Reference

Xiong, Liang, et al. "Temporal collaborative filtering with bayesian probabilistic tensor factorization."  *Proceedings of the 2010 SIAM international conference on data mining* . Society for Industrial and Applied Mathematics, 2010.
