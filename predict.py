# Import packages
import pandas as pd
import numpy as np

from scipy import optimize

import time

import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Set timer.
t1 = time.time()

# Load transaction data
df1 = pd.read_csv('Transaction_Data.csv')

# Pre-Process data 
df1['date'] = pd.to_datetime(df1['date'])
df1['date_no_year'] = [int(i) for i in df1['date'].dt.strftime("%m%d")]
df1['income'] = df1['quantity'] * df1['price']

# Load market data
df2 = pd.read_csv('Mandi_Characteristics.csv')

# Pre-Process data
df2.rename({'MandiId':'mandi_id'},axis=1,inplace=True)
df2['Type'] = df2['Type'].fillna(0)
df2['Type'] = df2['Type'].replace('Open Market',1)

# The PTF model based on Eq 3.9 in the paper
class PTF():
    
    def preprocess_data (self):
        cropId_max = np.max(np.array(df1['crop_id']))
        cropId_median = cropId_max // 2
        
        mandiId_max = np.max(np.array(df1['mandi_id']))
        mandiId_median = mandiId_max // 2

        # Divide dates into 2 zones.
        dates1 = np.array([int(i) for i in pd.date_range("2016-01-01", "2016-06-30").strftime("%m%d")])
        dates2 = np.array([int(i) for i in pd.date_range("2016-07-01", "2016-12-31").strftime("%m%d")])

        # Divide crops into 2 zones.
        crops1 = np.array(range(1, cropId_median)) 
        crops2 = np.array(range(cropId_median, cropId_max+1)) 

        # Divide mandis into 2 zones.
        mandis1 = np.array(range(1, mandiId_median)) 
        mandis2 = np.array(range(mandiId_median, mandiId_max+1)) 

        # Collect zones for dates, crops, mandis
        dates = [dates1, dates2]
        crops = [crops1, crops2]
        mandis = [mandis1, mandis2]
        
        return dates, crops, mandis

    def __init__(self):
        # Initialize parameters
        self.d = 2
        self.alpha = 0.1
        self.sigmaU = 2
        self.sigmaV = 2
        self.sigmaT = 2
        self.sigmaT0 = 2
        
        self.miuT0 = np.array([0]*self.d)
        self.dates, self.crops, self.mandis = self.preprocess_data ()

    # Use data before year 2019 as training set
    def preprocess_training_sets(self):
        
        train = df1.loc[df1['date'] < '2019-01-01'] 

        # Calculate weighted price
        train['total_income'] = train.groupby(['date_no_year', 'crop_id', 'mandi_id'])['income'].transform(lambda x: x.sum())
        train['total_quantity'] = train.groupby(['date_no_year', 'crop_id', 'mandi_id'])['quantity'].transform(lambda x: x.sum())
        train['weighted_price'] = train['total_income'] / train['total_quantity']

        # Drop duplicate data
        train = train.drop_duplicates(subset = ['date_no_year', 'crop_id', 'mandi_id'])[['date_no_year', 'crop_id', 'mandi_id', 'weighted_price']]

        for i in range(2):
            globals()[f'train{i+1}'] = train.loc[train['date_no_year'].isin(self.dates[i])]
            
        for j in range(1,3):
            globals()[f'train{j}_1'] = globals()[f'train{j}'].loc[globals()[f'train{j}']['crop_id'].isin(self.crops[0])]
            globals()[f'train{j}_2'] = globals()[f'train{j}'].loc[globals()[f'train{j}']['crop_id'].isin(self.crops[1])]
            
        for k in range(1, 3):
            
            for l in range(1, 3):
                temp = globals()[f'train{k}_{l}']
                
                globals()[f'train{k}_{l}_1'] = temp.loc[temp['mandi_id'].isin(self.mandis[0])].sort_values(by = ['date_no_year', 'crop_id', 'mandi_id'])
                globals()[f'train{k}_{l}_2'] = temp.loc[temp['mandi_id'].isin(self.mandis[1])].sort_values(by = ['date_no_year', 'crop_id', 'mandi_id'])

    # Use year 2019's data as testing set.
    def preprocess_testing_sets(self):
        test = df1.loc[df1['date'] >= '2019-01-01']

        # Calculate weighted price
        test['total_income'] = test.groupby(['date_no_year', 'crop_id', 'mandi_id'])['income'].transform(lambda x: x.sum())
        test['total_quantity'] = test.groupby(['date_no_year', 'crop_id', 'mandi_id'])['quantity'].transform(lambda x: x.sum())
        test['weighted_price'] = test['total_income'] / test['total_quantity']

        # Drop duplicates
        test = test.drop_duplicates(subset = ['date_no_year', 'crop_id', 'mandi_id'])[['date_no_year', 'crop_id', 'mandi_id', 'weighted_price']]

        for i in range(2):
            globals()[f'test{i+1}'] = test.loc[test['date_no_year'].isin(self.dates[i])]
            
        for j in range(1,3):
            globals()[f'test{j}_1'] = \
                globals()[f'test{j}'].loc[globals()[f'test{j}']['crop_id'].isin(self.crops[0])]
            
            globals()[f'test{j}_2'] = \
                globals()[f'test{j}'].loc[globals()[f'test{j}']['crop_id'].isin(self.crops[1])]
            
        for k in range(1, 3):
            
            for l in range(1, 3):
                temp = globals()[f'test{k}_{l}']
                
                globals()[f'test{k}_{l}_1'] = temp.loc[temp['mandi_id'].isin(self.mandis[0])].sort_values(by = ['date_no_year', 'crop_id', 'mandi_id'])
                globals()[f'test{k}_{l}_2'] = temp.loc[temp['mandi_id'].isin(self.mandis[1])].sort_values(by = ['date_no_year', 'crop_id', 'mandi_id'])

    # Construct matrix R in Eq 3.9 of the paper.
    def construct_R(self, train_or_test, quarterzone, cropzone, mandizone):
        dates, crops, mandis = self.preprocess_data ()

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

        # 0 stands for train, 1 stands for test
        if train_or_test == 0: 
            tmp_R = tmp_R.merge(globals()[f'train{quarterzone}_{cropzone}_{mandizone}'], how='left', on=['date_no_year', 'crop_id', 'mandi_id'])
        elif train_or_test == 1:
            tmp_R = tmp_R.merge(globals()[f'test{quarterzone}_{cropzone}_{mandizone}'], how='left', on=['date_no_year', 'crop_id', 'mandi_id'])
        
        self.R = np.array(tmp_R['weighted_price']).reshape(self.num_dates, self.num_crops, self.num_mandis)
        
        self.R_indi = np.where(np.isnan(self.R)==True, 0, 1)
        
        return self.R
    
    # Compute the log likelihood in Eq 3.9 of the paper
    def log_likelihood(self, x): 
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
        
        # Print the log likelihood in every optimization step
        # print(sum) 
        
        return sum
    
    def minimizer(self, **kwargs):
        
        minimum = optimize.minimize(self.log_likelihood, np.array([2]*(self.num_crops + self.num_mandis + self.num_dates + 1) * self.d).\
            reshape((self.num_crops + self.num_mandis + self.num_dates + 1) * self.d), **kwargs)
        
        return minimum
    
    # Calculate mean absolute error in PTF
    def MAE(self,test,minimum):
        
        x = minimum.x
        R_test_indi = np.where(np.isnan(test) == True, 0, 1)
        test = np.nan_to_num(test)

        if R_test_indi.sum() != 0:
            x = x.reshape(self.num_dates + self.num_crops + self.num_mandis + 1, self.d)
            
            U_star = x[0: self.num_crops]
            V_star = x[self.num_crops: self.num_crops + self.num_mandis]
            T_star = x[self.num_crops + self.num_mandis : self.num_crops + self.num_mandis + self.num_dates]
            
            arr = np.array(np.repeat(0, self.num_dates * self.num_crops * self.num_mandis).\
                    reshape(self.num_dates, self.num_crops, self.num_mandis))
            
            for i in range(self.d):
                t = T_star[:,i]
                u = U_star[:,i]
                v = V_star[:,i]
                
                tmp = (np.outer(t, u)[:, :, None] * v)
                tmp.shape
                
                arr = tmp + arr

            return (R_test_indi * abs(arr - test)).sum()/R_test_indi.sum()
        
        else:
            return None
        
# Implement a CART model
class CART():
    def __init__(self):
        # Load training data set
        train = df1.loc[df1['date'] < '2019-01-01']

        # Calculate weighted price
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

    def MAE(self, i, j, k):
        
        result = self.clf.predict(np.array(globals()[f'test{i}_{j}_{k}'][['date_no_year','crop_id', 'mandi_id', 'DistrictId','Lat','Lon','Type']]))
        
        error = np.mean(abs(np.array(globals()[f'test{i}_{j}_{k}']['weighted_price']) - result))
        
        return error

# Mean Model
class Mean() :
    def preprocess_data (self):
        # Load training data set
        train_for_mean = df1.loc[df1['date'] < '2019-01-01']

        # Calculate weighted price
        train_for_mean['total_income'] = train_for_mean.groupby(['crop_id', 'mandi_id'])['income'].transform(lambda x: x.sum())
        train_for_mean['total_quantity'] = train_for_mean.groupby(['crop_id', 'mandi_id'])['quantity'].transform(lambda x: x.sum())
        train_for_mean['weighted_price'] = train_for_mean['total_income'] / train_for_mean['total_quantity']

        # Calculate mean price
        mean_price = train_for_mean['income'].sum() / train_for_mean['quantity'].sum()

        # Drop duplicate
        train_for_mean = train_for_mean.drop_duplicates(subset=['crop_id','mandi_id'])[['crop_id', 'mandi_id', 'weighted_price']]
        
        return train_for_mean, mean_price

    def __init__(self):
        
        self.train_for_mean, self.mean_price = self.preprocess_data()

    # Calculate mean absolute error
    def MAE (self, i, j, k):
        tmp = globals()[f'test{i}_{j}_{k}'].merge(self.train_for_mean, how='left', on=['crop_id', 'mandi_id'])
        
        tmp['weighted_price_y'] = tmp['weighted_price_y'].fillna(self.mean_price)
        
        err = abs(tmp['weighted_price_x'] - tmp['weighted_price_y']).mean()
        
        return err

# Random Forest Model
class RF():
    def __init__(self, X ,y):
        
        self.X = X
        self.y = y
        
        self.regr = RandomForestRegressor(random_state=0).fit(self.X, self.y)
    
    def MAE (self, i, j, k):
        
        result = self.regr.predict(np.array(globals()[f'test{i}_{j}_{k}'][['date_no_year','crop_id', 'mandi_id', 'DistrictId','Lat','Lon','Type']]))
        
        err = np.mean(abs(np.array(globals()[f'test{i}_{j}_{k}']['weighted_price']) - result))
        
        return err

# Multi-Layer Perception Neural Network (CNN)
class CNN():
    def __init__(self, X, y):
        
        self.X = X
        self.y = y
        
        self.regr = MLPRegressor(random_state=1, max_iter=500).fit(self.X, self.y)
    
    def MAE (self,i,j,k):
        
        result = self.regr.predict(np.array(globals()[f'test{i}_{j}_{k}'][['date_no_year','crop_id', 'mandi_id', 'DistrictId','Lat','Lon','Type']]))
        
        err = np.mean(abs(np.array(globals()[f'test{i}_{j}_{k}']['weighted_price']) - result))
        
        return err

def main():
    
    # Construct PTF model
    ptf = PTF()
    ptf.preprocess_training_sets()
    ptf.preprocess_testing_sets()

    # Construct CART model
    cart = CART()
    cart.preprocess_testing_sets()
    cart.regression_tree()

    # Construct Mean model
    mean = Mean()

    # Construct a RF model with the same datasets in CART
    rf = RF(cart.X, cart.y)

    # Construct a Multi-Layer Perceptron model with the same datasets in CART
    cnn = CNN(cart.X, cart.y)

    # Initialize errors
    errs_Mean = []
    
    errs_PTF = []
    
    errs_CART = []
    
    errs_RF = []
    
    errs_CNN = []
    
    errs_total = []

    # Iterates through time zones
    for i in range(1,3): 
        
        # Iterates through crop zones
        for j in range(1,3): 
            
            # Iterates through mandi zones
            for k in range(1,3):
                
                if not globals()[f'test{i}_{j}_{k}'].empty:
                    
                    errs_Mean.append(mean.MAE (i, j ,k))

                    test = ptf.construct_R(1,i,j,k)
                    
                    ptf.construct_R(0,i,j,k)
                    
                    minimum = ptf.minimizer(method='SLSQP', options={'maxiter' : 18,'maxfev': 18})

                    # Calculate PTF error
                    errs_PTF.append(ptf.MAE(test,minimum))

                    # Calculate CART error
                    errs_CART.append(cart.MAE (i, j, k))

                    # Calculate RF error
                    errs_RF.append(rf.MAE(i,j,k))

                    # Calculate CNN error
                    errs_CNN.append(cnn.MAE(i,j,k))

                    # Calculate total error
                    errs_total.append(len(globals()[f'test{i}_{j}_{k}'].index))

    total = np.array(errs_total).sum()

    print("MAE using CART is",(np.array(errs_CART) * np.array(errs_total)).sum()/total,
        "\nMAE using PTF is",(np.array(errs_PTF)*np.array(errs_total)).sum()/total,
        "\nMAE using Mean is",(np.array(errs_Mean)*np.array(errs_total)).sum()/total,
        "\nMAE using random forest is", (np.array(errs_RF)*np.array(errs_total)).sum()/total,
        "\nMAE using multi-layer perceptron regressor is", (np.array(errs_CNN)*np.array(errs_total)).sum()/total)
    
    t2 = time.time()

    # Elapsed time
    print('Elapsed time is', t2-t1, 's.')
    
    # Visualize MAEs for different models
    names = list(range(len(errs_Mean)))
    
    x_axis = np.arange(len(names))
    
    plt.bar(x_axis - 0.4, errs_CART, width = 0.2, label = 'MAE_CART')
    
    plt.bar(x_axis - 0.2, errs_PTF, width = 0.2, label = 'MAE_PTF')
    
    plt.bar(x_axis + 0.0, errs_Mean, width = 0.2, label = 'MAE_Mean')
    
    plt.bar(x_axis + 0.2, errs_RF, width = 0.2, label = 'MAE_RF')
    
    plt.bar(x_axis + 0.4, errs_CNN, width = 0.2, label = 'MAE_CNN')
    
    plt.xticks(x_axis, names)
    
    plt.legend()
    
    plt.show()

# Start main()
if __name__ == "__main__":
    
    main()
