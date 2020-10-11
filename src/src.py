import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
plt.close('all')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('../data/raw/Historical Product Demand.csv')

df = df[['Product_Code', 'Warehouse', 'Date', 'Order_Demand']]

df['Product_Code'] = df['Product_Code'].str[-4:]
df['Warehouse'] = df['Warehouse'].str[-1:]

df = df[df.Order_Demand.str[0] != '(']
df['Order_Demand'] = df['Order_Demand'].astype(float)

df = df.sort_values(by=['Date'])

df = df.dropna()

dateSeries = pd.to_datetime(df['Date'], format='%Y/%m/%d', errors='coerce')
df['DayOfWeek'] = dateSeries.dt.dayofweek
df['DayOfMonth'] = dateSeries.dt.day
df['DayOfYear'] = dateSeries.dt.dayofyear
df['Week'] = dateSeries.dt.weekofyear
df['Month'] = dateSeries.dt.month
df['Year'] = dateSeries.dt.year
date = df.pop('Date')

df['Year'] = df.Year - 2011

df = pd.get_dummies(df, columns=['Warehouse'])

df.to_csv('../data/processed/product_demand_transformed.csv')

df['AbsoluteDay'] = df.DayOfYear + df.Year * 365

df[df['Product_Code'] == '1135'].plot.scatter(x='AbsoluteDay', y='Order_Demand', title='Product 1135 Demand by Day')
df[df['Product_Code'] == '1135'].plot.scatter(x='DayOfMonth', y='Order_Demand', title='Product 1135 Demand by Day of Month')
df[df['Product_Code'] == '1135'].plot.scatter(x='Week', y='Order_Demand', title='Product 1135 Demand by Week')
df[df['Product_Code'] == '1135'].plot.scatter(x='Month', y='Order_Demand', title='Product 1135 Demand by Month')
df[df['Product_Code'] == '1135'].plot.scatter(x='Year', y='Order_Demand', title='Product 1135 Demand by Year')

df[df['Product_Code'] == '1222'].plot.scatter(x='AbsoluteDay', y='Order_Demand', title='Product 1222 Demand by Day')
df[df['Product_Code'] == '1222'].plot.scatter(x='DayOfMonth', y='Order_Demand', title='Product 1222 Demand by Day of Month')
df[df['Product_Code'] == '1222'].plot.scatter(x='Week', y='Order_Demand', title='Product 1222 Demand by Week')
df[df['Product_Code'] == '1222'].plot.scatter(x='Month', y='Order_Demand', title='Product 1222 Demand by Month')
df[df['Product_Code'] == '1222'].plot.scatter(x='Year', y='Order_Demand', title='Product 1222 Demand by Year')

df[df['Product_Code'] == '1991'].plot.scatter(x='AbsoluteDay', y='Order_Demand', title='Product 1991 Demand by Day')
df[df['Product_Code'] == '1991'].plot.scatter(x='DayOfMonth', y='Order_Demand', title='Product 1991 Demand by Day of Month')
df[df['Product_Code'] == '1991'].plot.scatter(x='Week', y='Order_Demand', title='Product 1991 Demand by Week')
df[df['Product_Code'] == '1991'].plot.scatter(x='Month', y='Order_Demand', title='Product 1991 Demand by Month')
df[df['Product_Code'] == '1991'].plot.scatter(x='Year', y='Order_Demand', title='Product 1991 Demand by Year')

df1135 = df[df['Product_Code'] == '1135']
df1222 = df[df['Product_Code'] == '1222']
df1991 = df[df['Product_Code'] == '1991']

Y1135 = df1135.Order_Demand.values
Y1222 = df1222.Order_Demand.values
Y1991 = df1991.Order_Demand.values

X1135_all = df1135[['DayOfWeek', 'DayOfMonth', 'DayOfYear', 'Week', 'Month', 'Year', 'Warehouse_A', 'Warehouse_C', 'Warehouse_J', 'Warehouse_S']].values
X1135_week = df1135[['Week', 'Year', 'Warehouse_A', 'Warehouse_C', 'Warehouse_J', 'Warehouse_S']].values
X1135_month = df1135[['DayOfMonth', 'Month', 'Year', 'Warehouse_A', 'Warehouse_C', 'Warehouse_J', 'Warehouse_S']].values
X1135_day = df1135[['DayOfYear', 'Year', 'Warehouse_A', 'Warehouse_C', 'Warehouse_J', 'Warehouse_S']].values

X1222_all = df1222[['DayOfWeek', 'DayOfMonth', 'DayOfYear', 'Week', 'Month', 'Year', 'Warehouse_A', 'Warehouse_C', 'Warehouse_J', 'Warehouse_S']].values
X1222_week = df1222[['Week', 'Year', 'Warehouse_A', 'Warehouse_C', 'Warehouse_J', 'Warehouse_S']].values
X1222_month = df1222[['DayOfMonth', 'Month', 'Year', 'Warehouse_A', 'Warehouse_C', 'Warehouse_J', 'Warehouse_S']].values
X1222_day = df1222[['DayOfYear', 'Year', 'Warehouse_A', 'Warehouse_C', 'Warehouse_J', 'Warehouse_S']].values

X1991_all = df1991[['DayOfWeek', 'DayOfMonth', 'DayOfYear', 'Week', 'Month', 'Year', 'Warehouse_A', 'Warehouse_C', 'Warehouse_J', 'Warehouse_S']].values
X1991_week = df1991[['Week', 'Year', 'Warehouse_A', 'Warehouse_C', 'Warehouse_J', 'Warehouse_S']].values
X1991_month = df1991[['DayOfMonth', 'Month', 'Year', 'Warehouse_A', 'Warehouse_C', 'Warehouse_J', 'Warehouse_S']].values
X1991_day = df1991[['DayOfYear', 'Year', 'Warehouse_A', 'Warehouse_C', 'Warehouse_J', 'Warehouse_S']].values

tsplit = TimeSeriesSplit(n_splits=10)

r2_scores = []
for train_index, test_index in tsplit.split(X1135_month):
    X_train = X1135_month[train_index]
    Y_train = Y1135[train_index]
    
    X_test = X1135_month[test_index]
    Y_test = Y1135[test_index]
    
    predictions = GradientBoostingRegressor().fit(X_train, Y_train).predict(X_test)
    r2_scores.append(r2_score(Y_test, predictions))

print("Product 1135 Gradient Boosting With Week and Year Information")
print("r2_score: " + str(np.mean(r2_scores)))

r2_scores = []
for train_index, test_index in tsplit.split(X1222_all):
    X_train = X1222_all[train_index]
    Y_train = Y1222[train_index]
    
    X_test = X1222_all[test_index]
    Y_test = Y1222[test_index]
    
    predictions =MLPRegressor().fit(X_train, Y_train).predict(X_test)
    r2_scores.append(r2_score(Y_test, predictions))

print("Product 1222 MLP With All Time Information")
print("r2_score: " + str(np.mean(r2_scores)))

r2_scores = []
for train_index, test_index in tsplit.split(X1991_month):
    X_train = X1991_month[train_index]
    Y_train = Y1991[train_index]
    
    X_test = X1991_month[test_index]
    Y_test = Y1991[test_index]
    
    predictions = MLPRegressor().fit(X_train, Y_train).predict(X_test)
    r2_scores.append(r2_score(Y_test, predictions))

print("Product 1991 MLP With DayOfMonth, Month, and Year Information")
print("r2_score: " + str(np.mean(r2_scores)))
