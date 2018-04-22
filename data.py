import csv
import quandl
import numpy as np
import csv
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ochl
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
import os
import bs4 as bs


STOCK_COUNT = 0
TIME_PERIODS = 0
TICKERS = ["AAPL","MSFT","FB","GOOG","GOOGL","INTC","CSCO","NVDA","ORCL","IBM","ADBE","TXN","AVGO","CRM","QCOM","MU","AMAT","CTSH","INTU","EBAY","HPQ","ADI","LRCX","NOW","ADSK","RHT","HPE","WDC","MCHP","TWTR","CERN","NTAP","WDAY","MSI","SWKS","PANW","STX","XLNX","SYMC","CTL","KLAC","MXIM","DVMT","SPLK","ANSS","SNPS","CTXS","IAC","AKAM","ANET","CDW","CA","ON","CDNS","IT","VRSN","SSNC","FFIV","PTC","DOX","LDOS","MRVL","VMW","QRVO","GRUB","VEEV","AMD","TER","JNPR","CDK","FTNT","TYL","COMM","MSCC","ULTI","ZAYO","GRMN","GWRE","XRX","LOGM","MKSI","PFPT","AZPN","CY","BAH","ATHN","EPAM","CAVM","PAYC","DATA","BLKB","FICO","ENTG","Z","TDC","MPWR","ARRS","IDTI","NUAN","MDSO","VSM","SLAB","CREE","JCOM","LITE","CACI","CIEN","NCR","SAIC","VSAT","FEYE","ELLI","SNX","CVLT","MANH","ACIW","SMTC","TECD","IDCC","VRNT","SATS","CRUS","NTCT","ZG","VIAV","MDRX","PLT","CARS","PBI","FNSR","SYNA","EFII","DDD","P"]



style.use('ggplot')

start = dt.datetime(2010, 1, 1)
end = dt.datetime.now()

# df = web.DataReader("TSLA", 'morningstar', start, end)
# df.reset_index(inplace=True)
# df.set_index("Date", inplace=True)
# df = df.drop("Symbol", axis=1)
# print(df)
#print(df.head())
#df['Close'].plot()
#plt.show()


#os.makedirs('Desktop\stocks_dfs')
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

compileData() #Run to compile the data to one csv file