# EECS 731 Project 5: World Wide Products Inc.
Submission by Benjamin Wyss

## Project Overview

Examining historical product demand data to build time series forecasting models which predict the future demand for products

### Data Sets Used

Historical Product Demand - Taken from: https://www.kaggle.com/felixzhao/productdemandforecasting on 10/6/20

### Results

For three distinct products, I built and tested a variety of time series forecasting models using different time granularities to predict the demand of the product. Overall, no model for any product was able to achieve a positive average coefficient of determination in time series split testing. This is because the products that I had selected to build models for have irregular demand patterns and numerous outliers. All of the tested models appeared to struggle with the irregular patterns and outliers in the examined products' demands. It is likely that further data set cleaning to remove outlier values will produce better results for the products examined, but with the irregular patterns observed it will be very difficult to achieve high performance models.
