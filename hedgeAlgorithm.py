import RiskAdjustedReturnMetrics
import numpy as np
from numpy.linalg import inv
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd

# A.J. Imholte, Malini Sharma
# Algorithm Design and Analysis
# Shilad Sen
# Macalester College, Spring 2018

# A file with the implementation of the hedge algorithm.
# The file should be set to run...Simply run the file to see our results!

stocks = pd.read_csv('iShares_joined_closes.csv', index_col=0) # Stock closing prices
stockPriceChanges = pd.read_csv('iShares_Price_Changes.csv', index_col=0) # Stock price relatives - a number over 1 is an increase, under 1 is a decrease
stockReturns = pd.read_csv('iShares_percentage_change.csv', index_col=0) # Stock percentage changes over the time period
marketReturns = pd.read_csv('market.csv', index_col=0) # The returns for the S&P 500 and the iShares ETF

# Max changes for each period
maxChanges = []
for i in range(stockPriceChanges.shape[0]):
    maxChanges.append(max(stockPriceChanges.iloc[i,:]))


# Max percentage changes for each period
maxReturns = []
for j in range(stockReturns.shape[0]):
    maxReturns.append(max(stockReturns.iloc[j,:]))

STOCK_NUMBER = stockPriceChanges.shape[1] # The number of stocks we are considering
TIME_PERIODS = stockPriceChanges.shape[0] # The number of time periods we are running the algorithm on

#Initial weights for each stock:
stockWeights = [1.0/STOCK_NUMBER] * STOCK_NUMBER # The 'naive' weights...where each stock gets the same weight at the start

etfWeights2010 = [0.09344705900000001, 0.12388411800000002, 0.0042110202343750075, 0.0042110202343750075, 0.06978705900000001, 0.0042110202343750075, 0.07147705900000002, 0.004672353000000016, 0.047717647000000016, 0.08907294100000002, 0.009543529000000018, 0.017198235000000017, 0.0042110202343750075, 0.0036782350000000167, 0.0042110202343750075, 0.003976471000000016, 0.009841765000000018, 0.006859412000000016, 0.0052688240000000166, 0.0042110202343750075, 0.047518824000000015, 0.004871176000000016, 0.0023858820000000167, 0.0042110202343750075, 0.0029823530000000166, 0.0027835290000000168, 0.0042110202343750075, 0.004970588000000016, 0.0033800000000000162, 0.0042110202343750075, 0.0034794120000000168, 0.0054676470000000164, 0.0042110202343750075, 0.0042110202343750075, 0.0010935290000000165, 0.0042110202343750075, 0.004572941000000016, 0.004175294000000017, 0.007654706000000016, 0.0042110202343750075, 0.0029823530000000166, 0.003578824000000017, 0.0042110202343750075, 0.0042110202343750075, 0.0020876470000000167, 0.0016900000000000164, 0.004374118000000016, 0.0007952940000000166, 0.0021870590000000164, 0.0042110202343750075, 0.0042110202343750075, 0.0051694120000000164, 0.0015905880000000165, 0.0006958820000000166, 0.0007952940000000166, 0.0025847060000000165, 0.0042110202343750075, 0.0024852940000000164, 0.0042110202343750075, 0.0029823530000000166, 0.0024852940000000164, 0.004672353000000016, 0.0010935290000000165, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0022864710000000165, 0.0007952940000000166, 0.007356471000000016, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0005964710000000166, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.004473529000000017, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0007952940000000166, 0.0042110202343750075, 0.0003976470000000166, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0005964710000000166, 0.0042110202343750075, 0.0042110202343750075, 0.0027835290000000168, 0.0042110202343750075, 0.0006958820000000166, 0.0004970590000000166, 0.0014911760000000166, 0.0042110202343750075, 0.0042110202343750075, 0.0011929410000000166, 0.0032805880000000166, 0.0006958820000000166, 0.0042110202343750075, 0.0009941180000000165, 0.0004970590000000166, 0.0010935290000000165, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.00019882400000001657, 0.0005964710000000166, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0003976470000000166, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0006958820000000166, 0.0006958820000000166, 0.0042110202343750075, 0.0025847060000000165, 0.0042110202343750075, 0.0003976470000000166, 0.0003976470000000166, 0.0042110202343750075, 0.0042110202343750075]
# The "informed" weights from 2010 of the iShares technology etf.

# Initial Value for beta:
optimalBeta = math.sqrt(2.0 * math.log(STOCK_NUMBER) / TIME_PERIODS)
# This is based off of the original paper of the hedge algorithm. (0.0653)


# Plots the profits in terms of total wealth accumulated for a given series
def plotProfits(wealth=None):
    plt.cla()
    profit = np.zeros(shape=TIME_PERIODS)
    for i in range(STOCK_NUMBER):
        profit[0] = stockPriceChanges.iloc[i][0].tolist()
        for j in range(1, TIME_PERIODS):
            changes = stockPriceChanges.iloc[j-1][i].tolist()
            profit[j] = changes * profit[j-1]
        plt.plot(range(TIME_PERIODS), profit, linewidth = 0.35)
    if wealth:
        plt.plot(range(TIME_PERIODS), wealth, label='Hedge', linewidth = 2.75, color='tab:green')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.title('Total Wealth of Hedge Algorithm and iShares Stocks, 2010-2018')
    plt.xlabel('Period')
    plt.ylabel('Total Wealth')
    plt.grid(True)
    plt.show()
# plotProfits()

# Plots the prices of the stocks...used this to help visualize our data.
def plotPrices():
    plt.cla()
    for i in range(STOCK_NUMBER):
        prices = stocks.iloc[:,i]
        plt.plot(range(TIME_PERIODS+1), prices)
    plt.show()
# plotPrices()

def plotChanges():
    plt.cla()
    for i in range(STOCK_NUMBER):
        priceChanges = stockPriceChanges.iloc[:,i]
        plt.plot(range(TIME_PERIODS), priceChanges)
    plt.show()
# plotChanges()

# Plots the returns for each stock and for the Hedge algorithm if it is passed to the function.
def plotReturns(wealth=None):
    plt.cla()
    profit = np.zeros(TIME_PERIODS)
    for i in range(STOCK_NUMBER):
        profit[0] = stockReturns.iloc[i][0].tolist()
        for j in range(1, TIME_PERIODS):
            changes = stockReturns.iloc[j-1][i].tolist()
            profit[j] = changes + profit[j-1]
        plt.plot(range(TIME_PERIODS), profit, linewidth = 0.35)
        plt.grid(True)
    if wealth:
        plt.plot(range(TIME_PERIODS), wealth, label='Hedge', linewidth = 2.75, color='tab:green')
    plt.title('Hedge Algorithm Returns vs iShares Stock Returns, 2010-2018')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.xlabel('Period')
    plt.ylabel('Total Returns')
    plt.show()
# plotProfits()

# An implementation of the hedge algorithm
def hedge(weight = stockWeights, beta = optimalBeta):
    returns = []
    total = 1.0
    wealth = []
    returns.append(0.0)
    wealth.append(1.0)
    # Iterate over each time period:
    for i in range(TIME_PERIODS):
        allocation = [weight[j] * total / sum(weight) for j in range(STOCK_NUMBER)]
        # Choose allocations for each stock in this time period
        # print(allocation)
        priceVector = stockPriceChanges.iloc[i,:].tolist() # a list of stock price changes for this period
        # print(priceVector)

        #returns from this period
        total = np.dot(allocation, priceVector) # Dot product of the allocations by the price relaties returns the total wealth for this period.
        wealth.append(total)

        #new weight multipliers
        maxChange = maxChanges[i]
        difference = [maxChange - priceVector[n] for n in range(STOCK_NUMBER)] # Calculate the difference between the max change and the change for every stock
        weightUpdate = [beta ** difference[l] for l in range(STOCK_NUMBER)]

        #update weights
        weight = [weight[k] * weightUpdate[k] for k in range(STOCK_NUMBER)] # Reduce all weights other than for the stock that gained the most
        if min(weight) <= 0.001:
            weight = [weight[j] * 1000 for j in range(STOCK_NUMBER)]
            #print(stockWeights)


    print(np.mean(wealth)/np.std(wealth))
    plt.cla()
    wealth.pop(0)
    plotProfits(wealth)
    plt.show()

# A modified version of the hedge algorithm where we also calculate and display the results of the algorithm in terms of return.
def hedgeWithReturns(weight = stockWeights, beta = optimalBeta):

    # All the same steps as above other than adding returns.

    returns = []
    total = 1.0
    wealth = []
    returns.append(0.0)
    wealth.append(1.0)
    for i in range(TIME_PERIODS):
        allocation = [weight[j] * total / sum(weight) for j in range(STOCK_NUMBER)]
        # print(allocation)
        priceVector = stockPriceChanges.iloc[i,:].tolist()
        returnVector = stockReturns.iloc[i,:].tolist() # Percentages for each stock for this period
        # print(priceVector)

        #returns from this period

        total = np.dot(allocation, priceVector)
        wealth.append(total)
        returns.append(returns[i] + np.dot(allocation, returnVector)) # Adds the dot product of the allocation and the percentage changes to the return list

        #new weight multipliers
        maxChange = maxChanges[i]
        difference = [maxChange - priceVector[n] for n in range(STOCK_NUMBER)]
        weightUpdate = [beta ** difference[l] for l in range(STOCK_NUMBER)]

        #update weights
        weight = [weight[k] * weightUpdate[k] for k in range(STOCK_NUMBER)]
        if min(weight) <= 0.001:
            weight = [weight[j] * 1000 for j in range(STOCK_NUMBER)]
            #print(weight)
    print('Sharpe:') # A metric for risk-adjusted return
    print(RiskAdjustedReturnMetrics.sharpe_ratio(np.mean(returns),returns,0))
    print('Upside Potential') # Measures 'positive variance' -- how much the portfolio is posed to gain in the positive direction
    print(RiskAdjustedReturnMetrics.upside_potential_ratio(returns))
    print('Calmar:') # Returns adjusted for the maximum drawdown...the higher the number the better
    print(RiskAdjustedReturnMetrics.calmar_ratio(np.mean(returns), returns, 0))
    print('Volatility:') # Standard deviation of returns -- the simplest and most common form to measure risk
    print(RiskAdjustedReturnMetrics.vol(returns))
    marketReturn = marketReturns.iloc[:,0].tolist()
    print('Maximum Draw Down:') # Measures the maximum difference from a previous high to a new low -- want this number to be low
    print(RiskAdjustedReturnMetrics.max_dd(returns))
    print('Average Daily Return:')
    print(np.mean(returns))
    plt.cla()
    wealth.pop(0)
    plotProfits(wealth)
    plt.show()
    plt.cla()
    returns.pop(0)
    plotReturns(returns)
    plt.show()
    plt.cla()
    plotReturnsWithMarket(returns)
    plt.show()
    return returns

# A more compact version of the code above.
def hedgeWithReturnsNoGraphs(weight = stockWeights, beta = optimalBeta):
    returns = []
    total = 1.0
    wealth = []
    returns.append(0.0)
    wealth.append(1.0)
    for i in range(TIME_PERIODS):
        allocation = [weight[j] * total / sum(weight) for j in range(STOCK_NUMBER)]
        # print(allocation)
        priceVector = stockPriceChanges.iloc[i,:].tolist()
        returnVector = stockReturns.iloc[i,:].tolist()
        # print(priceVector)

        #returns from this period

        total = np.dot(allocation, priceVector)
        wealth.append(total)
        returns.append(returns[i] + np.dot(allocation, returnVector))

        #new weight multipliers
        maxChange = maxChanges[i]
        difference = [maxChange - priceVector[n] for n in range(STOCK_NUMBER)]
        weightUpdate = [beta ** difference[l] for l in range(STOCK_NUMBER)]

        #update weights
        weight = [weight[k] * weightUpdate[k] for k in range(STOCK_NUMBER)]
        if min(weight) <= 0.001:
            weight = [weight[j] * 1000 for j in range(STOCK_NUMBER)]
            #print(weight)
    return returns

# Plots the returns from the hedge algorithm with the market returns from the S&P 500 and the iShares ETF
def plotReturnsWithMarket(returns=None):
    plt.cla()
    profit = np.zeros(2089)
    labels = ['S&P 500', 'iShares Technology ETF']
    columns = marketReturns.shape[1]
    for i in range(columns):
        profit[0] = marketReturns.iloc[0,i].tolist()
        for j in range(1, marketReturns.shape[0]):
            print(j)
            changes = marketReturns.iloc[j-1,i].tolist()
            profit[j] = changes + profit[j-1]
        plt.plot(range(2088), profit[0:2088], label=labels[i] ,linewidth = 0.35)
        plt.grid(True)
    if returns:
        plt.plot(range(2088), returns[0:2088], label='Hedge', linewidth = 0.9, color='tab:green')
    plt.title('Hedge Algorithm Returns vs iShares Technology ETF & S&P 500 Returns, 2010-2018')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.xlabel('Period')
    plt.ylabel('Total Returns')
    plt.show()

# Basic market statistics used for benchmarking our results
def marketDiagnostics():
    sp500 = marketReturns.iloc[:,0].tolist()
    iShares = marketReturns.iloc[:,1].tolist()
    returns = []
    returns.append(sp500[0])
    for i in range(1,len(sp500)):
        returns.append(returns[i-1] + sp500[i-1])
    print('S&P500 diagnostics:')
    print('Sharpe:')
    print(RiskAdjustedReturnMetrics.sharpe_ratio(np.mean(returns),returns,0))
    print('Upside Potential')
    print(RiskAdjustedReturnMetrics.upside_potential_ratio(returns))
    print('Calmar:')
    print(RiskAdjustedReturnMetrics.calmar_ratio(np.mean(returns), returns, 0))
    print('Volatility:')
    print(RiskAdjustedReturnMetrics.vol(returns))
    marketReturn = marketReturns.iloc[:,0].tolist()
    print('Maximum Draw Down:')
    print(RiskAdjustedReturnMetrics.max_dd(returns))
    print('Average Daily Return:')
    print(np.mean(returns))
    returns = []
    returns.append(iShares[0])
    for j in range(1,len(iShares)):
        returns.append(returns[j-1] + iShares[j-1])
    print('iShares diagnostics:')
    print('Sharpe:')
    print(RiskAdjustedReturnMetrics.sharpe_ratio(np.mean(returns),returns,0))
    print('Upside Potential')
    print(RiskAdjustedReturnMetrics.upside_potential_ratio(returns))
    print('Calmar:')
    print(RiskAdjustedReturnMetrics.calmar_ratio(np.mean(returns), returns, 0))
    print('Volatility:')
    print(RiskAdjustedReturnMetrics.vol(returns))
    marketReturn = marketReturns.iloc[:,0].tolist()
    print('Maximum Draw Down:')
    print(RiskAdjustedReturnMetrics.max_dd(returns))
    print('Average Daily Return:')
    print(np.mean(returns))
# marketDiagnostics() uncomment to see the market statistics

# This function looks at our tuning parameter for beta and tries to operate it for both the informed and naive versions of our algorithm.
# This function can take a while to run, so we included the results below based off our data for sake of convenience.
def bestBeta():
    sharpeList = []
    for i in range(1,100):
        testBeta = i * 0.01
        print("Beta equals: " + str(testBeta))
        sharpeList.append(hedgeWithReturns(etfWeights2010, testBeta))
    print(str(max(sharpeList)))
    print(str(sharpeList.index(max(sharpeList))))
#bestBeta()
#This function takes a while to run...results are below:
#This returns a value of 0.16 for naive
#This returns a value of 0.12 for informed

# Plots all Hedge returns versuse the market returns for the naive, informed, naive optimized, and informed optimized hedge algorithms.
def plotComparisonsWithMarket(naiveReturns=None, informedReturns=None, naiveOptimizedReturns=None, informedOptimizedReturns=None):
    plt.cla()
    plt.grid(True)
    profit = np.zeros(2089)
    if naiveReturns:
        plt.plot(range(2088), naiveReturns[0:2088], label='Naive Hedge', linewidth = 0.35)
    if informedReturns:
        plt.plot(range(2088), informedReturns[0:2088], label='Informed Hedge', linewidth = 0.35)
    if naiveOptimizedReturns:
        plt.plot(range(2088), naiveOptimizedReturns[0:2088], label='Optimized Naive Hedge', linewidth = 0.35)
    if informedOptimizedReturns:
        plt.plot(range(2088), informedOptimizedReturns[0:2088], label='Optimized Informed Hedge', linewidth = 0.35)
    plt.title('Hedge Algorithm Returns vs iShares Technology ETF & S&P 500 Returns, 2010-2018')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.xlabel('Period')
    plt.ylabel('Total Returns')
    plt.show()

# A function meant to run all our alogorithm types to compare and constrast results for each
def compareHedgeAlgorithms():
    print('Gathering naive hedge results...')
    naiveReturns = hedgeWithReturnsNoGraphs()
    print('Gathering informed hedge results...')
    informedReturns = hedgeWithReturnsNoGraphs(etfWeights2010)
    print('Gathering naive hedge results...')
    naiveReturnsOptimized = hedgeWithReturnsNoGraphs(stockWeights, 0.16)
    print('Gathering informed hedge results...')
    informedOptimizedReturns = hedgeWithReturnsNoGraphs(etfWeights2010, 0.12)
    plotComparisonsWithMarket(naiveReturns, informedReturns, naiveReturnsOptimized, informedOptimizedReturns)
    print('Comparing and plotting the naive hedge to the market...')
    hedgeWithReturns()
    print('Comparing and plotting the informed hedge to the market...')
    hedgeWithReturns(etfWeights2010)
    print('Comparing and plotting the naive optimized hedge to the market...')
    hedgeWithReturns(stockWeights, 0.16)
    print('Comparing and plotting the informed optimized hedge to the market...')
    hedgeWithReturns(etfWeights2010, 0.12)

# The main function of our program. Reports the market diagnostics along with a comparison of each hedge algorithm to the market and other hedge algorithms.
def runProgram():
    marketDiagnostics()
    compareHedgeAlgorithms()

runProgram()

