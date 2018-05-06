import csv
import quandl
import numpy as np
import csv
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
import os
import bs4 as bs

# A.J. Imholte, Malini Sharma
# Algorithm Design and Analysis
# Shilad Sen
# Macalester College, Spring 2018

# This file handles all the data mining that was needed for this project, other than grabbing data for the S&P 500 and the iShares etf itself, which we grabbed from Yahoo! Finance.
# All other stock data comes from the Morningstar website and API, which we make use of in this file.

STOCK_COUNT = 101
TIME_PERIODS = 10830
TICKERS = ["AAPL","MSFT","FB","GOOG","GOOGL","INTC","CSCO","NVDA","ORCL","IBM","ADBE","TXN","AVGO","CRM","QCOM","MU","AMAT","CTSH","INTU","EBAY","HPQ","ADI","LRCX","NOW","ADSK","RHT","HPE","WDC","MCHP","TWTR","CERN","NTAP","WDAY","MSI","SWKS","PANW","STX","XLNX","SYMC","CTL","KLAC","MXIM","DVMT","SPLK","ANSS","SNPS","CTXS","IAC","AKAM","ANET","CDW","CA","ON","CDNS","IT","VRSN","SSNC","FFIV","PTC","DOX","LDOS","MRVL","VMW","QRVO","GRUB","VEEV","AMD","TER","JNPR","CDK","FTNT","TYL","COMM","MSCC","ULTI","ZAYO","GRMN","GWRE","XRX","LOGM","MKSI","PFPT","AZPN","CY","BAH","ATHN","EPAM","CAVM","PAYC","DATA","BLKB","FICO","ENTG","Z","TDC","MPWR","ARRS","IDTI","NUAN","MDSO","VSM","SLAB","CREE","JCOM","LITE","CACI","CIEN","NCR","SAIC","VSAT","FEYE","ELLI","SNX","CVLT","MANH","ACIW","SMTC","TECD","IDCC","VRNT","SATS","CRUS","NTCT","ZG","VIAV","MDRX","PLT","CARS","PBI","FNSR","SYNA","EFII","DDD","P"]
# The stock tickers of interest


style.use('ggplot')

start = dt.datetime(2010, 1, 1)
end = dt.datetime(2018, 4, 20)

#os.makedirs('Desktop\stocks_dfs') # sets up a directory to put the data

# Grabs data from the Morningstar API for each stock in our list of tickers and places them in independent csv files
def getMorningStarData():
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    for ticker in TICKERS:
        df = web.DataReader(ticker, 'morningstar', start, end)
        if(df.size == 10830):
            if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                print("Setting up csv for {}".format(ticker))
                df.reset_index(inplace=True)
                df.set_index("Date", inplace=True)
                df = df.drop("Symbol", axis=1)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            else:
                print('Already have {}'.format(ticker))
    print("Done!")

            # print(ticker)
            # df = web.DataReader(ticker, 'morningstar', start, end)
            # print(df['Close'])
            #print(df.size) #want a size of 10830

# getMorningStarData() #Uncomment to get all the data for iShares etf from Morningstar

# Joins our individual csvs into one with all stock price data from our time period for the stocks of interest
def compileData():
    mainDF = pd.DataFrame()
    for ticker in TICKERS:
        if os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            print("Joining data from ticker {}".format(ticker))
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
            df.rename(columns={'Close':ticker}, inplace=True)
            df.drop(['Open','High','Low','Volume'],1,inplace=True)
            if mainDF.empty:
                mainDF = df
            else:
                mainDF = mainDF.join(df, how='outer')
        else:
            print("Not joining data for {}".format(ticker))
    print(mainDF.head())
    mainDF.to_csv('iShares_joined_closes.csv')
    print("Done joining data!")

# A function we used to grab weights for each stock of interest based off its weight it had in the iShares etf in 2010.
def comepileStartingWeights():
    print('Getting the weights...')
    mainDF = pd.DataFrame()
    df = pd.read_csv('iShares_weights_2010', index_col=0)
    weights = []
    for ticker in TICKERS:
        if(ticker in df.index):
            temp = df.loc[ticker].Weight
            if df.loc[ticker] is not None:
                weights.append(temp)
        else:
            weights.append(0)
    rest = 1.0-sum(weights)
    zeroCount = weights.count(0)
    for i in range(len(weights)):
        if weights[i] == 0:
            weights[i] = rest/zeroCount
    newSum = 1.0-sum(weights)
    for j in range(len(weights)):
        weights[j] = weights[j] - (newSum/len(weights))
    print(sum(weights))
    return weights


print(comepileStartingWeights())

#compileData() #Run to compile the data to one csv file


# Some more useful code we used that doesn't need to be ran right now
'''
dates = allData['Date']
# print(dates[1:2166])

# priceChangeData = pd.DataFrame(columns=allData.keys())
# for i in range(1, allData.shape[0]):
#     priceChangeData.loc[i] = [all]


diff = pd.DataFrame(data=dates.iloc[1:2166])
diff.set_index('Date', inplace=True)

print(allData.shape)
print(allData.values)

#For each column
    #For each row in the column

for i in range(1, allData.shape[1]):
    dataToAdd = []
    for j in range(1,allData.shape[0]):
        oldPrice = allData.iloc[j-1, i]
        newPrice = allData.iloc[j,i]
        change = newPrice/oldPrice
        dataToAdd.append(change)
    diff[allData.keys()[i]] = pd.Series(dataToAdd, index=diff.index)
print(diff.head)

maxDiff = pd.DataFrame(data=dates.iloc[1:2166])
maxDiff.set_index('Date', inplace=True)
maxData = []
for n in range(diff.shape[0]):
    maxPrice = max(diff.iloc[n,:])
    maxData.append(maxPrice)

maxDiff = pd.Series(maxData)
print(maxDiff.head())
maxDiff.to_csv('iShares Max Changes.csv')

for i in range(1, allData.shape[1]):
    dataToAdd = []
    for j in range(1,allData.shape[0]):
        oldPrice = allData.iloc[j-1, i]
        newPrice = allData.iloc[j,i]
        change = (newPrice-oldPrice)/oldPrice
        dataToAdd.append(change)
    diff[allData.keys()[i]] = pd.Series(dataToAdd, index=diff.index)
print(diff.head)

# diff.to_csv('iShares_percentage_change.csv')



# maxPrices = []
# for n in range(diff.shape[0]):
#     maxPrices.append(max(diff[n]))



# for i in range(1, allData.shape[1]):
#     for j in range(2, allData.shape[0]):
#         oldPrice = allData.iloc[i,(j-1)]
#         newPrice = allData.iloc[i,j]
#     diff[allData.keys()[i]] = pd.Series((newPrice - oldPrice), index=diff.index)
# print(diff.head)



# diff.to_csv('iShares_Price_Changes.csv')

'''



