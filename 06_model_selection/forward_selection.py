import matplotlib.pyplot as plt
import numpy as np
import itertools
from numpy import random as rd
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

num_data = 1000
x1 = rd.random(size = num_data)
x1 = x1 / x1.std()

x2 = rd.random(size = num_data)
x2 = x2 / x2.std()

x3 = x1*x2
x3 = x3 / x3.std()

y = x1 + x2 + x3

x4 = rd.random(size = num_data)
x5 = rd.random(size = num_data)
x6 = rd.random(size = num_data)

df = pd.DataFrame({
    'x1':x1,
    'x2':x2,
    'x3':x3,
    'x4':x4,
    'x5':x5,
    'x6':x6,
    'y':y
})

df_train, df_test = train_test_split(df,  test_size=0.3)

def features_candidates(select_bag, already_selected, forward: bool = True):
    if forward:
        for new_feature_candidate in select_bag:
            features = [new_feature_candidate] + already_selected
            yield features



def select_best_regresor(select_bag, already_selected):
    results = pd.DataFrame()
    for new_feature_candidate in select_bag:
        features = [new_feature_candidate] + already_selected

        model = LinearRegression()
        model.fit(df_train[features], df_train['y'])

        predictions = model.predict(df_test[features])
        mse = ((predictions - df_test['y']) ** 2).sum()

        df_to_concat = pd.DataFrame({
            'features': [features],
            'mse': [mse],
            'new_feature': [new_feature_candidate]
        })

        results = pd.concat([results, df_to_concat])

    results = results.reset_index(drop=True)
    best_index = results['mse'].argmin()

    new_feature_selected = results.loc[best_index,'new_feature']
    best_mse = results.loc[best_index, 'mse']

    return new_feature_selected, best_mse


original_features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
selected_ones = []
mse_list = []
for j in range(len(original_features)):
    best_regresor, mse = select_best_regresor(original_features, selected_ones)

    mse_list.append(mse)
    original_features.remove(best_regresor)
    selected_ones += [best_regresor]
    print(selected_ones)

plt.plot(
    range(len(mse_list)),
    mse_list,
    '-o'
)
plt.show()