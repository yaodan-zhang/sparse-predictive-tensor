import pandas as pd
import numpy as np

df1 = pd.read_csv('Transaction Data Full.csv')
df2 = pd.read_csv('Updated Mandi Characteristics.csv')
df3 = pd.read_csv('Crop Characteristics.csv')

df1['date'] = pd.to_datetime(df1['date'])

#correlation of prices between geographically closed markets
districts = np.array(df2.drop_duplicates(['DistrictId'])['DistrictId']) #get all districts of the mandis

n = 0
sum_corr = 0

for i in districts:
    cur_district = df2.loc[df2['DistrictId'] == i]['MandiId']
    transaction = df1.loc[df1['mandi_id'].isin(cur_district)]

    if not transaction.empty:
        crop = transaction['crop_id'].mode() #select the most popularly traded crop in this district
        transaction = transaction.loc[transaction['crop_id'] == crop[0]][['date','mandi_id','price']] #select the most traded crop for analysis
        most_mandi = transaction.groupby('mandi_id').apply(lambda x: len(x)).sort_values() #get mandis most traded of this crop

        if most_mandi.size > 2:
            most_mandi = np.array(most_mandi.index)
            mandi1 = most_mandi[-1]
            mandi2 = most_mandi[-2]
            m1 = transaction.loc[transaction['mandi_id'] == mandi1].drop_duplicates(['date'])
            m2 = transaction.loc[transaction['mandi_id'] == mandi2].drop_duplicates(['date'])
            table_m1_m2 = m1.merge(m2, on=['date'])

            if not table_m1_m2.empty:
                correlation = table_m1_m2['price_x'].corr(table_m1_m2['price_y'])

                if not np.isnan(correlation):
                    print('The correlation between mandi ',mandi1,' and mandi ',mandi2,' over crop ',crop[0],' is ',correlation,'.',sep='')
                    n += 1
                    sum_corr += correlation

print('The average correlation over',n,'couples of near markets is',sum_corr/n,'.')

#market's variability
mandis = np.array(df1.drop_duplicates(['mandi_id'])['mandi_id'])
variances = np.array([])

for j in mandis:
    transaction = df1.loc[df1['mandi_id'] == j]
    #get the most traded crop of this mandis to do variability analysis
    most_crop = np.array(transaction.groupby('crop_id').apply(lambda x: len(x)).sort_values().index) 
    crop_id = most_crop[-1]
    transaction = transaction.loc[transaction['crop_id'] == crop_id]
    transaction['income'] = transaction['quantity'] * transaction['price']
    transaction['total_quantity'] = transaction.groupby(['date'])['quantity'].transform(lambda x: x.sum())
    transaction['total_income'] = transaction.groupby(['date'])['income'].transform(lambda x: x.sum())
    transaction['weighted_price'] = transaction['total_income']/transaction['total_quantity']
    transaction = transaction.drop_duplicates(['date'])

    if transaction['weighted_price'].size >= 50:
        variance = transaction['weighted_price'].var()
        variances = np.append(variances, variance, axis=None)
        print(variance)

print('Minimum of variances',np.min(variances),\
    '\n25th quantile of variances',np.quantile(variances,.25),
    '\nMedian quantile of variances',np.quantile(variances,.50),
    '\n75th quantile of variances',np.quantile(variances,.75),
    '\nMaximum of variances',np.max(variances))

#checking influential market
influens = np.array([])
market_id = np.array([])

for k in mandis:
    this_market = df1.loc[df1['mandi_id'] == k]

    #crop traded for most days of this market
    crop = np.array(this_market.drop_duplicates(['date','crop_id']).groupby('crop_id').apply(lambda x: len(x)).sort_values().index)[-1]
    this_market = this_market.loc[this_market['crop_id'] == crop][['date','price']].drop_duplicates(['date'])
    this_market.rename(columns = {'price':'weighted_price'}, inplace = True)
    this_market['date'] = this_market.date + pd.Timedelta(days=1) #add 1 day to do influencial analysis with other markets

    other_markets = df1.loc[df1['mandi_id'] != k]
    other_markets = other_markets.loc[other_markets['crop_id'] == crop][['date','quantity','price']]
    other_markets['income'] = other_markets['quantity'] * other_markets['price']
    other_markets['total_quantity'] = other_markets.groupby(['date'])['quantity'].transform(lambda x: x.sum())
    other_markets['total_income'] = other_markets.groupby(['date'])['income'].transform(lambda x: x.sum())
    other_markets['weighted_price'] = other_markets['total_income']/other_markets['total_quantity']
    other_markets = other_markets.drop_duplicates(['date'])[['date','weighted_price']]

    table_this_others = this_market.merge(other_markets, on=['date'])

    if table_this_others['date'].size >= 500:
        corr = table_this_others['weighted_price_x'].corr(table_this_others['weighted_price_y'])
        influens = np.append(influens, corr, axis=None)
        market_id = np.append(market_id, k, axis = None)
        print('The correlation table over crop',crop,'\'s weighted price between market',k,' and other markets is',sep='')
        print(table_this_others)
        print('Correlation equals ',corr,'.',sep = '')
        print()

print('The maximum correlations obtained above is ',influens[np.argmax(influens)],', by market', market_id[np.argmax(influens)],'.',sep='')

#checking correlation across crops
crops = np.array(df1.drop_duplicates(['crop_id'])['crop_id'])

df1['crops_income'] = df1['price'] * df1['quantity']
df1['crops_total_quantity'] = df1.groupby(['date','crop_id'])['quantity'].transform(lambda x: x.sum())
df1['crops_total_income'] = df1.groupby(['date','crop_id'])['crops_income'].transform(lambda x: x.sum())
df1['crops_weighted_price'] = df1['crops_total_income']/df1['crops_total_quantity']

df1_copy = df1.drop_duplicates(['date','crop_id'])[['date','crop_id','crops_weighted_price']]

ind = np.array(df1_copy.groupby('crop_id').apply(lambda x: len(x)).sort_values().index)[-5:]

most_traded_crops = np.array(crops[ind]) #get the most traded 5 crops interms of trading days
most_traded_crops = np.sort(most_traded_crops)

for m in most_traded_crops:
    crop_1 = df1_copy.loc[df1_copy['crop_id'] == m][['date','crops_weighted_price']]
    
    for n in most_traded_crops:
        if n > m:
            crop_2 = df1_copy.loc[df1_copy['crop_id'] == n][['date','crops_weighted_price']]
            table_crop1_crop2 = crop_1.merge(crop_2, on=['date'])
            if table_crop1_crop2['date'].size >= 200:
                corr = table_crop1_crop2['crops_weighted_price_x'].corr(table_crop1_crop2['crops_weighted_price_y'])
                print('The correlation of crop',m,' and crop',n,' over price is ',corr,'.',sep='')
                print(table_crop1_crop2)
                print()
