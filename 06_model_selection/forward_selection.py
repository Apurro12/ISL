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


def features_candidates(decreasing_bag: list, increasing_bag: list, forward: bool = True, first_iter = True):

    if forward is True:
        for new_feature_candidate in decreasing_bag:
            features = [new_feature_candidate] + increasing_bag
            yield features, new_feature_candidate

    # This is not fully functional yet
    if forward is False:
        if first_iter:
            yield decreasing_bag, []

        if not first_iter:
            for remove_feature_candidate in decreasing_bag:
                new_feature_set = decreasing_bag[:]
                new_feature_set.remove(remove_feature_candidate)
                yield new_feature_set, remove_feature_candidate

def select_best_regresor(decreasing_bag, increasing_bag, forward, first_iter):

    results = pd.DataFrame()
    for features, changed_feature in features_candidates(decreasing_bag, increasing_bag, forward, first_iter):

        model = LinearRegression()
        model.fit(df_train[features], df_train['y'])

        predictions = model.predict(df_test[features])
        mse = ((predictions - df_test['y']) ** 2).sum()

        df_to_concat = pd.DataFrame({
            'features': [features],
            'mse': [mse],
            'changed_feature': [changed_feature]
        })

        results = pd.concat([results, df_to_concat])

        if changed_feature == []:
            break

    results = results.reset_index(drop=True)
    best_index = results['mse'].argmin()

    new_feature_selected = results.loc[best_index,'changed_feature']
    best_mse = results.loc[best_index, 'mse']

    return new_feature_selected, best_mse

def update_features(best_regresor, decreasing_bag, increasing_bag):

    if best_regresor == []:
        return decreasing_bag, []

    decreasing_bag.remove(best_regresor)
    increasing_bag += [best_regresor]

    return decreasing_bag, increasing_bag


def select_features(forward: bool = True):
    decreasing_bag = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    num_features = list(range(len(decreasing_bag)))
    increasing_bag = []
    mse_list = []
    xticks = []
    first_iter = True
    for _ in num_features:
        updated_regresor, mse = select_best_regresor(
            decreasing_bag,
            increasing_bag,
            forward,
            first_iter #Just by backward
        )

        decreasing_bag, increasing_bag = update_features(updated_regresor, decreasing_bag, increasing_bag)

        mse_list.append(mse)

        xticks.append(str(updated_regresor))
        print(increasing_bag)
        first_iter = False

    plt.tight_layout()
    plt.plot(
        range(len(mse_list)),
        mse_list,
        '-o'
    )
    plt.xticks(
        range(len(mse_list)),
        xticks,
        rotation=-90
    )
    plt.show()

if __name__ == '__main__':
    select_features(forward = True)
    select_features(forward=False)