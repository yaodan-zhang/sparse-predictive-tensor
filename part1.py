import pandas as pd
import numpy as np

# Load the datasets.
df1 = pd.read_csv('Transaction_Data.csv')
df2 = pd.read_csv('Mandi_Characteristics.csv')
df3 = pd.read_csv('Crop_Characteristics.csv')
df1['date'] = pd.to_datetime(df1['date'])

# EDA1: Examine price correlation in geographically close markets.
def PriceCorrOverCloseMarkets ():
    # get all mandi districts
    districts = np.array(df2.drop_duplicates(['DistrictId'])['DistrictId'])
    n = 0
    sum_corr = 0

    for i in districts:
        mandis = df2[df2['DistrictId'] == i]['MandiId']
        transactions = df1[df1['mandi_id'].isin(mandis)]

        if not transactions.empty:
            # select the most traded crop in this district
            crop = transactions['crop_id'].mode()
            transactions = transactions[transactions['crop_id'] == crop[0]][['date','mandi_id','price']]
            # get mandis most traded of this crop
            most_mandi = transactions.groupby('mandi_id').apply(lambda x: len(x)).sort_values()

            if most_mandi.size > 2:
                most_mandi = np.array(most_mandi.index)
                # get the top 2 mandis trading the crop most
                mandi1 = most_mandi[-1]
                mandi2 = most_mandi[-2]
                m1 = transactions[transactions['mandi_id'] == mandi1].drop_duplicates(['date'])
                m2 = transactions[transactions['mandi_id'] == mandi2].drop_duplicates(['date'])
                # merge the transaction data according to same dates
                table_m1_m2 = m1.merge(m2, on=['date'])

                if not table_m1_m2.empty:
                    corr = table_m1_m2['price_x'].corr(table_m1_m2['price_y'])

                    if not np.isnan(corr):
                        print('Correlation between mandi ',mandi1,' and mandi ',mandi2,' over crop ',crop[0],' is ',corr,'.',sep='')
                        n += 1
                        sum_corr += corr

    print('The average price correlation over ', n,' pairs of close markets is ', sum_corr/n,'.', sep = '')

#EDA2: Market's variability.
def MarketVariability ():
    mandis = np.array(df1.drop_duplicates(['mandi_id'])['mandi_id'])
    variances = np.array([])

    for j in mandis:
        transactions = df1[df1['mandi_id'] == j]
        # get the most traded crop of this market
        most_crop = np.array(transactions.groupby('crop_id').apply(lambda x: len(x)).sort_values().index) 
        crop_id = most_crop[-1]

        # calculate the daily weighted price of the crop
        crop_transac = transactions[transactions['crop_id'] == crop_id]
        crop_transac['income'] = crop_transac['quantity'] * crop_transac['price']
        crop_transac['total_quantity'] = crop_transac.groupby(['date'])['quantity'].transform(lambda x: x.sum())
        crop_transac['total_income'] = crop_transac.groupby(['date'])['income'].transform(lambda x: x.sum())
        crop_transac['weighted_price'] = crop_transac['total_income']/crop_transac['total_quantity']
        crop_transac = crop_transac.drop_duplicates(['date'])

        # Set a threshold for computing price variance
        if crop_transac['weighted_price'].size >= 50:
            variance = crop_transac['weighted_price'].var()
            variances = np.append(variances, variance, axis=None)

    print('Of all markets, the minimum price variance for a single market is ', np.min(variances),\
        ', the 25th quantile price variance is ', np.quantile(variances,.25),
        ', median price variance is ', np.quantile(variances,.50),
        ', the 75th quantile price variance is ', np.quantile(variances,.75),
        ', the maximum price variance is ', np.max(variances), ".", sep = '')

#EDA3: Determine an influential market.
def InfluentialMarket ():
    influens = np.array([])
    market_id = np.array([])
    mandis = np.array(df1.drop_duplicates(['mandi_id'])['mandi_id'])
    for k in mandis:
        this_market = df1.loc[df1['mandi_id'] == k]

        # Select the most traded crop in this market.
        crop = np.array(this_market.drop_duplicates(['date','crop_id']).groupby('crop_id').apply(lambda x: len(x)).sort_values().index)[-1]
        this_market = this_market.loc[this_market['crop_id'] == crop][['date','price']].drop_duplicates(['date'])
        this_market.rename(columns = {'price':'weighted_price'}, inplace = True)
        # Add 1 day to do influencial analysis on other markets.
        this_market['date'] = this_market.date + pd.Timedelta(days = 1)

        other_markets = df1.loc[df1['mandi_id'] != k]
        # Calculate the daily weighted price of this crop in other markets.
        other_markets = other_markets.loc[other_markets['crop_id'] == crop][['date','quantity','price']]
        other_markets['income'] = other_markets['quantity'] * other_markets['price']
        other_markets['total_quantity'] = other_markets.groupby(['date'])['quantity'].transform(lambda x: x.sum())
        other_markets['total_income'] = other_markets.groupby(['date'])['income'].transform(lambda x: x.sum())
        other_markets['weighted_price'] = other_markets['total_income']/other_markets['total_quantity']
        other_markets = other_markets.drop_duplicates(['date'])[['date','weighted_price']]
        # Select transactions that happens on intersected dates.
        table_this_others = this_market.merge(other_markets, on=['date'])

        if table_this_others['date'].size >= 500: # Set a threshold.
            corr = table_this_others['weighted_price_x'].corr(table_this_others['weighted_price_y'])
            influens = np.append(influens, corr, axis = None)
            market_id = np.append(market_id, k, axis = None)

            print('The price correlation table over crop ', crop,' between market ', k,' and other markets is\n',\
                table_this_others, "\n", 
                'Correlation equals ', corr, '.\n', sep = '')

    print('The maximum correlations obtained above is ', influens[np.argmax(influens)],', by market ', int(market_id[np.argmax(influens)]),'.',sep='')

#EDA4: Correlation across most traded crops.
def CropCorrelation ():
    crops = np.array(df1.drop_duplicates(['crop_id'])['crop_id'])

    # Calculate the daily weighted price for each crop.
    df1['crops_income'] = df1['price'] * df1['quantity']
    df1['crops_total_quantity'] = df1.groupby(['date','crop_id'])['quantity'].transform(lambda x: x.sum())
    df1['crops_total_income'] = df1.groupby(['date','crop_id'])['crops_income'].transform(lambda x: x.sum())
    df1['crops_weighted_price'] = df1['crops_total_income']/df1['crops_total_quantity']
    df1_copy = df1.drop_duplicates(['date','crop_id'])[['date','crop_id','crops_weighted_price']]

    # Get the most traded 5 crops interms of trading days.
    ind = np.array(df1_copy.groupby('crop_id').apply(lambda x: len(x)).sort_values().index)[-5:]
    most_traded_crops = np.sort(np.array(crops[ind]))

    for m in most_traded_crops:
        crop_1 = df1_copy.loc[df1_copy['crop_id'] == m][['date','crops_weighted_price']]
        
        for n in most_traded_crops:
            if n > m:
                crop_2 = df1_copy.loc[df1_copy['crop_id'] == n][['date','crops_weighted_price']]
                table_crop1_crop2 = crop_1.merge(crop_2, on=['date'])
                if table_crop1_crop2['date'].size >= 400:
                    corr = table_crop1_crop2['crops_weighted_price_x'].corr(table_crop1_crop2['crops_weighted_price_y'])
                    print('Price correlation between crop', m, ' and crop', n, ' is ', corr, '.',
                        "\n", table_crop1_crop2, "\n", sep='')


if __name__ == "__main__":
    PriceCorrOverCloseMarkets()
    MarketVariability ()
    InfluentialMarket ()
    CropCorrelation ()