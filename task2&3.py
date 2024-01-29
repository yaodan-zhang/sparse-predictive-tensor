import pandas as pd
import numpy as np
from scipy import optimize
import time
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

t1 = time.time()

errs_Mean = []
errs_PTF = []
errs_CART = []
errs_Mean_num = []
errs_PTF_num = []
errs_CART_num = []

df1 = pd.read_csv('Transaction Data Full.csv')
df1['date'] = pd.to_datetime(df1['date'])
df1['date_no_year'] = [int(i) for i in df1['date'].dt.strftime("%m%d")]
df1['income'] = df1['quantity'] * df1['price']
df2 = pd.read_csv('Updated Mandi Characteristics.csv')
df2.rename({'MandiId':'mandi_id'},axis=1,inplace=True)
df2['Type'] = df2['Type'].fillna(0)
df2['Type'] = df2['Type'].replace('Open Market',1)

cropId_max = np.max(np.array(df1['crop_id']))
cropId_median = cropId_max // 2
mandiId_max = np.max(np.array(df1['mandi_id']))
mandiId_median = mandiId_max // 2

#divide dates into 2 zones
dates1 = np.array([int(i) for i in pd.date_range("2016-01-01", "2016-06-30").strftime("%m%d")])
dates2 = np.array([int(i) for i in pd.date_range("2016-07-01", "2016-12-31").strftime("%m%d")])

#divide crops into 2 zones
crops1 = np.array(range(1, cropId_median)) 
crops2 = np.array(range(cropId_median, cropId_max+1)) 

#divide mandis into 2 zones
mandis1 = np.array(range(1, mandiId_median)) 
mandis2 = np.array(range(mandiId_median, mandiId_max+1)) 

#collect zones for dates, crops, mandis
dates = [dates1, dates2]
crops = [crops1, crops2]
mandis = [mandis1, mandis2]

class PTF():
    def __init__(self):
        self.d = 2
        self.alpha = 0.1
        self.sigmaU = 2
        self.sigmaV = 2
        self.sigmaT = 2
        self.sigmaT0 = 2
        self.miuT0 = np.array([0]*self.d)

    def preprocess_training_sets(self):    
        train = df1.loc[df1['date'] < '2019-01-01'] #use years before 2019 as training set
        train['total_income'] = train.groupby(['date_no_year', 'crop_id', 'mandi_id'])['income'].transform(lambda x: x.sum())
        train['total_quantity'] = train.groupby(['date_no_year', 'crop_id', 'mandi_id'])['quantity'].transform(lambda x: x.sum())
        train['weighted_price'] = train['total_income'] / train['total_quantity']
        train = train.drop_duplicates(subset = ['date_no_year', 'crop_id', 'mandi_id'])[['date_no_year', 'crop_id', 'mandi_id', 'weighted_price']]
        for i in range(2):
            globals()[f'train{i+1}'] = train.loc[train['date_no_year'].isin(dates[i])]
        for j in range(1,3):
            globals()[f'train{j}_1'] = globals()[f'train{j}'].loc[globals()[f'train{j}']['crop_id'].isin(crops[0])]
            globals()[f'train{j}_2'] = globals()[f'train{j}'].loc[globals()[f'train{j}']['crop_id'].isin(crops[1])]
        for k in range(1, 3):
            for l in range(1, 3):
                temp = globals()[f'train{k}_{l}']
                globals()[f'train{k}_{l}_1'] = temp.loc[temp['mandi_id'].isin(mandis[0])].sort_values(by = ['date_no_year', 'crop_id', 'mandi_id'])
                globals()[f'train{k}_{l}_2'] = temp.loc[temp['mandi_id'].isin(mandis[1])].sort_values(by = ['date_no_year', 'crop_id', 'mandi_id'])

    def preprocess_testing_sets(self):
        test = df1.loc[df1['date'] >= '2019-01-01']
        test['total_income'] = test.groupby(['date_no_year', 'crop_id', 'mandi_id'])['income'].transform(lambda x: x.sum())
        test['total_quantity'] = test.groupby(['date_no_year', 'crop_id', 'mandi_id'])['quantity'].transform(lambda x: x.sum())
        test['weighted_price'] = test['total_income'] / test['total_quantity']
        test = test.drop_duplicates(subset = ['date_no_year', 'crop_id', 'mandi_id'])[['date_no_year', 'crop_id', 'mandi_id', 'weighted_price']]
        for i in range(2):
            globals()[f'test{i+1}'] = test.loc[test['date_no_year'].isin(dates[i])]
        for j in range(1,3):
            globals()[f'test{j}_1'] = \
                globals()[f'test{j}'].loc[globals()[f'test{j}']['crop_id'].isin(crops[0])]
            globals()[f'test{j}_2'] = \
                globals()[f'test{j}'].loc[globals()[f'test{j}']['crop_id'].isin(crops[1])]
        for k in range(1, 3):
            for l in range(1, 3):
                temp = globals()[f'test{k}_{l}']
                globals()[f'test{k}_{l}_1'] = temp.loc[temp['mandi_id'].isin(mandis[0])].sort_values(by = ['date_no_year', 'crop_id', 'mandi_id'])
                globals()[f'test{k}_{l}_2'] = temp.loc[temp['mandi_id'].isin(mandis[1])].sort_values(by = ['date_no_year', 'crop_id', 'mandi_id'])

    def construct_R(self, train_or_test, quarterzone, cropzone, mandizone): #construct the observed R in Eq. 3.9 of the paper
        date = dates[quarterzone-1]
        crop = crops[cropzone-1]
        mandi = mandis[mandizone-1]
        self.num_dates = date.size; self.num_crops = crop.size; self.num_mandis = mandi.size
        tmp1 = np.repeat(date, self.num_crops * self.num_mandis)
        tmp2 = np.array([np.repeat(crop, self.num_mandis)] * self.num_dates).flatten()
        tmp3 = np.array([mandi] * self.num_dates * self.num_crops).flatten()
        tmp_R = pd.DataFrame()
        tmp_R['date_no_year'] = tmp1
        tmp_R['crop_id'] = tmp2
        tmp_R['mandi_id'] = tmp3
        if train_or_test == 0: #0 stands for train, 1 stands for test
            tmp_R = tmp_R.merge(globals()[f'train{quarterzone}_{cropzone}_{mandizone}'], how='left', on=['date_no_year', 'crop_id', 'mandi_id'])
        elif train_or_test == 1:
            tmp_R = tmp_R.merge(globals()[f'test{quarterzone}_{cropzone}_{mandizone}'], how='left', on=['date_no_year', 'crop_id', 'mandi_id'])
        self.R = np.array(tmp_R['weighted_price']).reshape(self.num_dates, self.num_crops, self.num_mandis)
        self.R_indi = np.where(np.isnan(self.R)==True, 0, 1)
        return self.R
    
    def log_likelihood(self, x): #compute the log_likelihood in Eq. 3.9 of the paper
        sum = 0
        x = x.reshape(self.num_dates + self.num_crops + self.num_mandis + 1, self.d)
        U = x[0: self.num_crops]
        V = x[self.num_crops: self.num_mandis+self.num_crops]
        T = x[self.num_mandis+self.num_crops: self.num_mandis+self.num_crops+self.num_dates]
        T0 = x[-1]
        arr = np.array(np.repeat(0, self.num_dates*self.num_crops*self.num_mandis).\
            reshape(self.num_dates, self.num_crops, self.num_mandis))
        for i in range(self.d):
            t = T[:,i]
            u = U[:,i]
            v = V[:,i]
            tmp = (np.outer(t, u)[:, :, None] * v)
            tmp.shape
            arr = tmp + arr
        self.R_copy = np.nan_to_num(self.R)
        sum += ((self.R_indi * (self.R_copy-arr))**2).sum()
        sum += np.square(U).sum()/(self.alpha * self.sigmaU**2 * 2)
        sum += np.square(V).sum()/(self.alpha * self.sigmaV**2 * 2)
        sum += (((T[1:]-T[:-1])**2).sum() + ((T[0] - T0)**2).sum())/(self.alpha * self.sigmaT**2 * 2)
        sum += ((T0-self.miuT0)**2).sum()/(self.alpha * self.sigmaT0**2 * 2)
        print(sum) #print the log likelihood in every optimization step
        return sum
    
    def minimizer(self, **kwargs):
        minimum = optimize.minimize(self.log_likelihood, np.array([2]*(self.num_crops + self.num_mandis + self.num_dates + 1) * self.d).\
            reshape((self.num_crops + self.num_mandis + self.num_dates + 1) * self.d), **kwargs)
        return minimum

class CART():
    def __init__(self):
        train = df1.loc[df1['date'] < '2019-01-01']
        train['total_income'] = train.groupby(['date_no_year','crop_id', 'mandi_id'])['income'].transform(lambda x: x.sum())
        train['total_quantity'] = train.groupby(['date_no_year','crop_id', 'mandi_id'])['quantity'].transform(lambda x: x.sum())
        train['weighted_price'] = train['total_income'] / train['total_quantity']
        train = train.drop_duplicates(subset = ['date_no_year','crop_id', 'mandi_id'])
        train = train.merge(df2[['mandi_id','DistrictId','Lat','Lon','Type']], how='left', on=['mandi_id'])
        self.train = train.fillna(0)

    def preprocess_testing_sets(self):
        for i in range(1,3):
            for j in range(1, 3):
                for k in range(1, 3):
                    globals()[f'test{i}_{j}_{k}'] = globals()[f'test{i}_{j}_{k}'].merge(df2[['mandi_id','DistrictId','Lat','Lon','Type']], how='left', on=['mandi_id']).fillna(0)

    def regression_tree(self):
        self.X = np.array(self.train[['date_no_year','crop_id','mandi_id','DistrictId','Lat','Lon','Type']])
        self.y = np.array(self.train['weighted_price'])
        model = tree.DecisionTreeRegressor()
        self.clf = model.fit(self.X, self.y)

    def predict_tree(self,i,j,k):
        if not globals()[f'test{i}_{j}_{k}'].empty:
            result = self.clf.predict(np.array(globals()[f'test{i}_{j}_{k}'][['date_no_year','crop_id', 'mandi_id', 'DistrictId','Lat','Lon','Type']]))
            error = np.mean(abs(np.array(globals()[f'test{i}_{j}_{k}']['weighted_price']) - result))
            return error
        else:
            return None

def main():
    #error generated by using mean
    def calculate_error_mean(train_for_mean, mean_price,i,j,k):
        if not globals()[f'test{i}_{j}_{k}'].empty:
            tmp = globals()[f'test{i}_{j}_{k}'].merge(train_for_mean, how='left', on=['crop_id', 'mandi_id'])
            tmp['weighted_price_y'] = tmp['weighted_price_y'].fillna(mean_price)
            err = abs(tmp['weighted_price_x']-tmp['weighted_price_y']).mean()
            return err
        else:
            return None
        
    #error generated by PTF
    def calculate_error_PTF(test, minimum):
        R_test_indi = np.where(np.isnan(test)==True, 0, 1)
        test = np.nan_to_num(test)

        if R_test_indi.sum() != 0:
            minimum = minimum.reshape(ptf.num_dates + ptf.num_crops + ptf.num_mandis + 1, ptf.d)
            U_star = minimum[0: ptf.num_crops]
            V_star = minimum[ptf.num_crops: ptf.num_crops + ptf.num_mandis]
            T_star = minimum[ptf.num_crops+ptf.num_mandis : ptf.num_crops+ptf.num_mandis+ptf.num_dates]
            arr = np.array(np.repeat(0, ptf.num_dates * ptf.num_crops * ptf.num_mandis).\
                    reshape(ptf.num_dates, ptf.num_crops, ptf.num_mandis))
            
            for i in range(ptf.d):
                t = T_star[:,i]
                u = U_star[:,i]
                v = V_star[:,i]
                tmp = (np.outer(t, u)[:, :, None] * v)
                tmp.shape
                arr = tmp + arr

            return (R_test_indi * abs(arr - test)).sum()/R_test_indi.sum()
        else:
            return None
        
    ptf = PTF()
    ptf.preprocess_training_sets()
    ptf.preprocess_testing_sets()

    cart = CART()
    cart.preprocess_testing_sets()
    cart.regression_tree()

    train_for_mean = df1.loc[df1['date'] < '2019-01-01']
    train_for_mean['total_income'] = train_for_mean.groupby(['crop_id', 'mandi_id'])['income'].transform(lambda x: x.sum())
    train_for_mean['total_quantity'] = train_for_mean.groupby(['crop_id', 'mandi_id'])['quantity'].transform(lambda x: x.sum())
    train_for_mean['weighted_price'] = train_for_mean['total_income'] / train_for_mean['total_quantity']

    mean_price = train_for_mean['income'].sum() / train_for_mean['quantity'].sum()
    train_for_mean = train_for_mean.drop_duplicates(subset=['crop_id','mandi_id'])[['crop_id', 'mandi_id', 'weighted_price']]

    for i in range(1,3): #iterates through time zones
        for j in range(1,3): #iterates through crop zones
            for k in range(1,3): #iterates through mandi zones
                test = ptf.construct_R(1,i,j,k)
                ptf.construct_R(0,i,j,k)
                minimum = ptf.minimizer(method='SLSQP', options={'maxiter' : 18,'maxfev': 18})
                err_Mean = calculate_error_mean(train_for_mean, mean_price, i, j ,k )
                err_PTF = calculate_error_PTF(test, minimum.x)
                err_CART = cart.predict_tree(i,j,k)
                if not globals()[f'test{i}_{j}_{k}'].empty:#if test is not empty
                    errs_Mean.append(err_Mean)
                    errs_Mean_num.append(len(globals()[f'test{i}_{j}_{k}'].index))
                    errs_PTF.append(err_PTF)
                    errs_PTF_num.append(len(globals()[f'test{i}_{j}_{k}'].index))
                    errs_CART.append(err_CART)
                    errs_CART_num.append(len(globals()[f'test{i}_{j}_{k}'].index))

    print("Error of price per day per crop per mandi generated using CART is",(np.array(errs_CART)*np.array(errs_CART_num)).sum()/np.array(errs_CART_num).sum(),
        "\nError of price per day per crop per mandi generated using PTF is",(np.array(errs_PTF)*np.array(errs_PTF_num)).sum()/np.array(errs_PTF_num).sum(),
        "\nError or price per day per crop per mandi generated using Mean is",(np.array(errs_Mean)*np.array(errs_Mean_num)).sum()/np.array(errs_Mean_num).sum())
    
    #build Random Forest using the same data sets as in CART
    regr = RandomForestRegressor(random_state=0)
    regr = regr.fit(cart.X, cart.y)
    errs_RF = []
    errs_RF_num = []
    for i in range(1,3):
        for j in range(1,3):
            for k in range(1,3):
                if not globals()[f'test{i}_{j}_{k}'].empty:
                    result = regr.predict(np.array(globals()[f'test{i}_{j}_{k}'][['date_no_year','crop_id', 'mandi_id', 'DistrictId','Lat','Lon','Type']]))
                    error = np.mean(abs(np.array(globals()[f'test{i}_{j}_{k}']['weighted_price']) - result))
                    errs_RF.append(error)
                    errs_RF_num.append(len(globals()[f'test{i}_{j}_{k}'].index))
    
    print("Error or price per day per crop per mandi generated using Random Forest is",(np.array(errs_RF) * np.array(errs_RF_num)).sum()/np.array(errs_RF_num).sum())
    
    t2 = time.time()
    
    print('Elapsed time is', t2-t1, 's.')
    
    #plot the errors made by 4 models
    names = list(range(len(errs_Mean)))
    x_axis = np.arange(len(names))
    
    plt.bar(x_axis -0.3, errs_CART, width=0.2, label = 'error using CART')
    plt.bar(x_axis -0.1, errs_PTF, width=0.2, label = 'error using PTF')
    plt.bar(x_axis +0.1, errs_Mean, width=0.2, label = 'error using Mean')
    plt.bar(x_axis +0.3, errs_RF, width=0.2, label = 'error using RandomForest')
    
    plt.xticks(x_axis, names)
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()