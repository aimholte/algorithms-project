import data
import numpy as np
import math
import matplotlib.pyplot as plt



# Step one: initialize weights for each stock and value for beta:
weight = [1.0/data.STOCK_NUMBER]*data.STOCK_NUMBER

# Value for beta:
beta = math.sqrt(2.0 * math.log(data.STOCK_NUMBER) / data.TIME_PERIODS)

# Step two: implement the Hedge algorithm:
def hedge(weight, beta):
    returns = list()
    returns.append(0.0)



