import pandas as pd
import numpy as np# basic imports
import matplotlib.pyplot as plt
import seaborn as sns
import os

fashion = pd.read_csv('fashion.csv')

fashion = pd.DataFrame(fashion)

#define how to aggregate various fields
agg_functions = {'BrandName': 'first', 'Contempt': 'sum'}

#create new DataFrame by combining rows with same id values
df_new = fashion.groupby(fashion['UserId']).aggregate(agg_functions)

#view new DataFrame
print(df_new)