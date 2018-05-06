import RiskAdjustedReturnMetrics
import numpy as np
from numpy.linalg import inv
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd

stocks = pd.read_csv('iShares_joined_closes.csv', index_col=0)
stockPriceChanges = pd.read_csv('iShares_Price_Changes.csv', index_col=0)
stockReturns = pd.read_csv('iShares_percentage_change.csv', index_col=0)
marketReturns = pd.read_csv('market.csv', index_col=0)


# get a row as  a list: print(stockPriceChanges.iloc[0,:].tolist())
maxChanges = []
for i in range(stockPriceChanges.shape[0]):
    maxChanges.append(max(stockPriceChanges.iloc[i,:]))

# Max percentage changes for each periodL
maxReturns = []
for j in range(stockReturns.shape[0]):
    maxReturns.append(max(stockReturns.iloc[i,:]))

STOCK_NUMBER = stockPriceChanges.shape[1] # The number of stocks we are considering
TIME_PERIODS = stockPriceChanges.shape[0] # The number of time periods we are running the algorithm on

#Initial weights for each stock:

stockWeights = [1.0/STOCK_NUMBER] * STOCK_NUMBER # The 'naive' weights...where each stock gets the same weight at the start

etfWeights2010 = [0.09344705900000001, 0.12388411800000002, 0.0042110202343750075, 0.0042110202343750075, 0.06978705900000001, 0.0042110202343750075, 0.07147705900000002, 0.004672353000000016, 0.047717647000000016, 0.08907294100000002, 0.009543529000000018, 0.017198235000000017, 0.0042110202343750075, 0.0036782350000000167, 0.0042110202343750075, 0.003976471000000016, 0.009841765000000018, 0.006859412000000016, 0.0052688240000000166, 0.0042110202343750075, 0.047518824000000015, 0.004871176000000016, 0.0023858820000000167, 0.0042110202343750075, 0.0029823530000000166, 0.0027835290000000168, 0.0042110202343750075, 0.004970588000000016, 0.0033800000000000162, 0.0042110202343750075, 0.0034794120000000168, 0.0054676470000000164, 0.0042110202343750075, 0.0042110202343750075, 0.0010935290000000165, 0.0042110202343750075, 0.004572941000000016, 0.004175294000000017, 0.007654706000000016, 0.0042110202343750075, 0.0029823530000000166, 0.003578824000000017, 0.0042110202343750075, 0.0042110202343750075, 0.0020876470000000167, 0.0016900000000000164, 0.004374118000000016, 0.0007952940000000166, 0.0021870590000000164, 0.0042110202343750075, 0.0042110202343750075, 0.0051694120000000164, 0.0015905880000000165, 0.0006958820000000166, 0.0007952940000000166, 0.0025847060000000165, 0.0042110202343750075, 0.0024852940000000164, 0.0042110202343750075, 0.0029823530000000166, 0.0024852940000000164, 0.004672353000000016, 0.0010935290000000165, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0022864710000000165, 0.0007952940000000166, 0.007356471000000016, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0005964710000000166, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.004473529000000017, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0007952940000000166, 0.0042110202343750075, 0.0003976470000000166, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0005964710000000166, 0.0042110202343750075, 0.0042110202343750075, 0.0027835290000000168, 0.0042110202343750075, 0.0006958820000000166, 0.0004970590000000166, 0.0014911760000000166, 0.0042110202343750075, 0.0042110202343750075, 0.0011929410000000166, 0.0032805880000000166, 0.0006958820000000166, 0.0042110202343750075, 0.0009941180000000165, 0.0004970590000000166, 0.0010935290000000165, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.00019882400000001657, 0.0005964710000000166, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0003976470000000166, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0042110202343750075, 0.0006958820000000166, 0.0006958820000000166, 0.0042110202343750075, 0.0025847060000000165, 0.0042110202343750075, 0.0003976470000000166, 0.0003976470000000166, 0.0042110202343750075, 0.0042110202343750075]
# The "informed" weights from 2010 of the iShares technology etf.

# Value for beta:
optimalBeta = math.sqrt(2.0 * math.log(STOCK_NUMBER) / TIME_PERIODS)

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

def plotProfitsV2(timePeriods,wealth=None):
    plt.cla()
    profit = np.zeros(shape=timePeriods)
    for i in range(STOCK_NUMBER):
        profit[0] = stockPriceChanges.iloc[i][0].tolist()
        for j in range(1, timePeriods):
            changes = stockPriceChanges.iloc[j-1][i].tolist()
            profit[j] = changes * profit[j-1]
        plt.plot(range(timePeriods), profit, linewidth = 0.35)
    if wealth:
        plt.plot(range(timePeriods), wealth, label='Hedge', linewidth = 2.75, color='tab:green')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.xlabel('Period')
    plt.ylabel('Total Wealth')
    plt.show()

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

def hedge(weight = stockWeights, beta = optimalBeta):
    returns = []
    total = 1.0
    wealth = []
    returns.append(0.0)
    wealth.append(1.0)
    for i in range(TIME_PERIODS):
        allocation = [weight[j] * total / sum(weight) for j in range(STOCK_NUMBER)]
        # print(allocation)
        priceVector = stockPriceChanges.iloc[i,:].tolist()
        # print(priceVector)

        #returns from this period

        total = np.dot(allocation, priceVector)
        wealth.append(total)

        #new weight multipliers
        maxChange = maxChanges[i]
        difference = [maxChange - priceVector[n] for n in range(STOCK_NUMBER)]
        weightUpdate = [beta ** difference[l] for l in range(STOCK_NUMBER)]

        #update weights
        weight = [weight[k] * weightUpdate[k] for k in range(STOCK_NUMBER)]
        if min(weight) <= 0.001:
            weight = [weight[j] * 1000 for j in range(STOCK_NUMBER)]
        #print(stockWeights)


    print(np.mean(wealth)/np.std(wealth))
    plt.cla()
    wealth.pop(0)
    plotProfits(wealth)
    plt.show()


def hedgeV2(trainingPeriods, testingPeriods, weight = stockWeights, beta = optimalBeta,):
    returns = []
    total = 1.0
    wealth = []
    returns.append(0.0)
    wealth.append(1.0)
    for i in range(trainingPeriods):
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

    #reset for testing:
    total = 1.0
    testWealth = []
    testReturns = []
    testReturns.append(0.0)
    for n in range(testingPeriods):
        allocation = [weight[j] * total / sum(weight) for j in range(STOCK_NUMBER)]
        # print(allocation)
        priceVector = stockPriceChanges.iloc[n,:].tolist()
        # print(priceVector)
        returnVector = stockReturns.iloc[n,:].tolist()

        #returns from this period
        total = np.dot(allocation, priceVector)
        testWealth.append(total)
        testReturns.append(testReturns[n] + np.dot(allocation, returnVector))

    # print(weight)
    # plt.cla()
    # wealth.pop(0)
    # plotProfitsV2(testingPeriods, testWealth)
    # plt.show()
    return [RiskAdjustedReturnMetrics.sharpe_ratio(np.mean(testReturns), testReturns, 0), RiskAdjustedReturnMetrics.upside_potential_ratio(testReturns),RiskAdjustedReturnMetrics.max_dd(testReturns), RiskAdjustedReturnMetrics.calmar_ratio(np.mean(testReturns), testReturns, 0), np.mean(testReturns), RiskAdjustedReturnMetrics.vol(testReturns)]


def hedgeWithReturns(weight = stockWeights, beta = optimalBeta):
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


def performTesting(numSimulations):
    sharpeRatios = []
    for i in range(2, numSimulations, 10):
        training = i
        testing = TIME_PERIODS-training
        print('Training periods='+str(training)+' ,testing periods='+str(testing), ' Sharpe ratio='+str(hedgeV2(training, testing, etfWeights2010)))
        sharpeRatios.append(hedgeV2(training, testing, etfWeights2010))
    print(str(max(sharpeRatios)), str(sharpeRatios.index(max(sharpeRatios))))

# performTesting(2165) # this takes a long time to run so here are the results: best sharpe ratio: 4.59, with 993 training periods and 1172 testing

# Optimal training size for this problem is 1069 periods!

# benchmarking:

moreTrain = 1069
moreTest = stockPriceChanges.shape[0] - moreTrain
# marketDiagnostics()
# print(hedge(etfWeights2010))
# hedge()
#plotReturns(wealth=None)
#hedgeWithReturns(etfWeights2010)
# performTesting(2165)
#hedgeWithReturns()

