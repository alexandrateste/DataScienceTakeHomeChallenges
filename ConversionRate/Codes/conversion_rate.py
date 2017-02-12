import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os
from sklearn.cross_validation import train_test_split
# from treeinterpreter import treeinterpreter as ti
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

plt.ion()


def data_extraction(file):

    dframe = pd.read_csv(file)
    dframe = dframe[dframe.age < 85.]
    # Very unlikely that people older than 85 years old browse the web
    # -- actually, the outliers were > 100 years old
    print "Dimensions: ", dframe.shape
    print("Summary: %s" %dframe.describe())
    for col in dframe.columns:
        if col not in ['country', 'source']:
            print("Non empty rows for %s: %s" %(col, np.sum(np.isfinite(dframe[col].ravel()))))
        else:
            subset = dframe[dframe[col] != '']
            print "Non empty rows for ", col, ':', subset.shape[0]
    # No data are missing

    return dframe


def investigation(df0):
    dframe = df0.copy()
    corr_matrix = dframe.corr()
    print("Correlation matrix:")
    print(corr_matrix)
    # No high correlations, as expected
    # Start seeing possible relationship between # pages visited and converted (i.e. the outcome)
    country = dframe.groupby('country')['age'].count()
    source = dframe.groupby('source')['age'].count()

    print("Country distribution:\n %s" %country)
    print("Source distribution:\n %s" %source)

    filename = out_dir_plot + 'age_distribution.png'
    if not os.path.exists(filename):
        plt.figure(100)
        dframe['age'].plot(kind='hist', bins=range(0,85,5))
        plt.ylabel('Frequency')
        plt.xlabel('Age')
        plt.title("Age distribution")
        plt.savefig(filename)
        plt.close()

    filename = out_dir_plot + 'conversion_rate_age.png'
    if not os.path.exists(filename):
        plt.figure(100)
        plt.plot(dframe['age'], dframe['converted'], marker='o', linestyle='None')
        plt.ylabel('Converted')
        plt.xlabel('Age')
        plt.title("Converted vs. age")
        plt.savefig(filename)
        plt.close()


    filename = out_dir_plot + 'conversion_rate_pages.png'
    if not os.path.exists(filename):
        plt.figure(100)
        plt.plot(dframe['total_pages_visited'], dframe['converted'], marker='o', linestyle='None')
        plt.ylabel('Converted')
        plt.xlabel('Total pages visited')
        plt.title("Converted vs. pages visited")
        plt.savefig(filename)
        plt.close()


def age_groups(df0):

    dframe = df0.copy()
    dframe['age_group'] = ''
    dframe.ix[dframe['age'] < 15.,'age_group'] = '<15'
    dframe.ix[(dframe['age'] >= 15.) & (dframe['age'] < 25.),'age_group'] = '15-24'
    dframe.ix[(dframe['age'] >= 25.) & (dframe['age'] < 35.),'age_group'] = '25-34'
    dframe.ix[(dframe['age'] >= 35.) & (dframe['age'] < 45.),'age_group'] = '35-44'
    dframe.ix[dframe['age'] >= 45.,'age_group'] = '>=45'

    return dframe


def conversions_per_group(df0, element, metric):
    dframe = df0.copy()
    conversion_group = []
    list_of_elements = list(set(dframe[element]))
    for elem in list_of_elements:
        subset = dframe[dframe[element] == elem]
        conversions = 100. * subset[metric].sum() / subset.shape[0]
        conversion_group.append(conversions)
    conversion_df = pd.DataFrame(conversion_group, columns=['values'])
    conversion_df['index'] = list_of_elements
    conversion_df = conversion_df.set_index('index')
    conversion_df = conversion_df.sort_values(['values'], ascending=False)

    if metric == 'converted':
        tittle = 'Conversion rate'
    elif metric == 'new_user':
        tittle = 'Percentage new users'

    filename = out_dir_plot + tittle.lower().replace(' ', '_') + '_' + element + '.png'
    if not os.path.exists(filename):
        plt.figure(200)
        conversion_df.plot(kind='bar', legend=False)
        plt.ylabel(tittle + ' (%)')
        plt.title(tittle + " by " + element)
        plt.savefig(filename)
        plt.close()

    return conversion_df


def current_conversion_rates(df0):
    dframe = df0.copy()
    total_conversions = 100.*dframe['converted'].sum() / dframe.shape[0]
    print("Total conversion rate: %s %%" %round(total_conversions, 2))
    conversion_country_df = conversions_per_group(dframe, 'country', 'converted')
    conversion_source_df = conversions_per_group(dframe, 'source', 'converted')
    conversion_agegroup_df = conversions_per_group(dframe, 'age_group', 'converted')
    conversion_user_df = conversions_per_group(dframe, 'new_user', 'converted')
    newuser_country_df = conversions_per_group(dframe, 'country', 'new_user')
    newuser_source_df = conversions_per_group(dframe, 'source', 'new_user')
    newuser_agegroup_df = conversions_per_group(dframe, 'age_group', 'new_user')

    return conversion_country_df, conversion_source_df, \
           conversion_agegroup_df, conversion_user_df, \
           newuser_country_df, newuser_source_df, \
           newuser_agegroup_df


def pages(df0, element, operation):

    dframe = df0.copy()

    pages_group = []
    list_of_elements = list(set(dframe[element]))
    for elem in list_of_elements:
        subset = dframe[dframe[element] == elem]
        if operation == 'mean':
            pages_aggr = subset['total_pages_visited'].mean()
        elif operation == "median":
            pages_aggr = subset['total_pages_visited'].median()
        pages_group.append(pages_aggr)
    pages_aggr_df = pd.DataFrame(pages_group, columns=['values'])
    pages_aggr_df['index'] = list_of_elements
    pages_aggr_df = pages_aggr_df.set_index('index')
    pages_aggr_df = pages_aggr_df.sort_values(['values'], ascending=False)

    filename = out_dir_plot + operation + '_total_pages_visited_' + element + '.png'
    if not os.path.exists(filename):
        plt.figure(200)
        pages_aggr_df.plot(kind='bar', legend=False)
        plt.ylabel(operation.capitalize() + ' Total pages visited')
        plt.title(operation.capitalize() + ' Total pages visited by ' + element)
        plt.savefig(filename)
        plt.close()

    return pages_aggr_df


def current_pages_visited(df0, operation):
    dframe = df0.copy()
    aggr_pages_country = pages(dframe, 'country', operation)
    aggr_pages_source = pages(dframe, 'source', operation)
    aggr_pages_agegroup = pages(dframe, 'age_group', operation)

    return aggr_pages_country, aggr_pages_source, aggr_pages_agegroup


def train_test(df0):
    dframe = df0.copy()
    X = dframe.drop(['converted'], axis=1).as_matrix()
    y = np.asarray(dframe['converted'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)

    return X_train, X_test, y_train, y_test


def data_preparation(df0):
    dframe = df0.copy()
    dframe = dframe.drop(['age_group'], axis=1)

    # Normalizes the numeric values that are not binary
    dframe['age'] = preprocessing.scale(dframe['age'])
    dframe['total_pages_visited'] = preprocessing.scale(dframe['total_pages_visited'])

    # Creates pseudo-dummy variables for categorical data
    dframe_extended = pd.get_dummies(dframe, prefix=['country', 'source'])

    return dframe_extended


def model_building(df0, pred_model):

    dframe = df0.copy()
    dframe = data_preparation(dframe)

    X_train, X_test, y_train, y_test = train_test(dframe)

    print("Building model %s" %pred_model)
    if pred_model == 'Random Forest':
        modl = RandomForestClassifier()
    elif pred_model == 'Random Forest grid search':
        parameters_grid = {"n_estimators": [10, 35, 60, 85],
                      "max_depth": [None, 4, 5, 6],
                      "max_features": [5, 6, 7, 8, 9, 10, 11,],
                      "min_samples_split": [2, 3, 6, 8, 10, 12],
                      "min_samples_leaf": [1, 2, 3, 4, 5],
                      "bootstrap": [True],
                      "oob_score": [False],
                      "criterion": ['gini', 'entropy']}

        modl = GridSearchCV(RandomForestClassifier(), param_grid=parameters_grid, cv=5)  # Finds the best paramaters
        modl.fit(X_train, y_train)
        modl.predict(X_test)

        print("Best parameters set found :\n %s\n" %modl.best_params_)
        print("Grid scores on development set:")
        means = modl.cv_results_['mean_test_score']
        stds = modl.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, modl.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
    elif pred_model == 'SVM Linear':
        modl = SVC(kernel='linear')
    elif pred_model == 'SVM RBF':
        modl = SVC(kernel='rbf')
    elif pred_model == 'Logistic Regression':
        modl = LogisticRegression()
    elif pred_model == 'kNN':
        ref_score = -99
        nbr_neighbors = -9
        for k in range(2,10):
            print("In loop for %s neighbors" %k)
            modl = KNeighborsClassifier(n_neighbors=k)
            modl.fit(X_train, y_train)
            modl.predict(X_test)
            if modl.score(X_test, y_test) > ref_score:
                ref_score = modl.score(X_test, y_test)
                nbr_neighbors = k

        modl = KNeighborsClassifier(n_neighbors=nbr_neighbors)
        print("%s is the optimal number of neighbors" %nbr_neighbors)

    modl.fit(X_train, y_train)
    modl.predict(X_test)

    if pred_model == 'Random Forest':
        labels = list(dframe.columns.values)
        labels.remove('converted')
        feat_importance = pd.DataFrame(modl.feature_importances_, columns=['values'])
        feat_importance['index'] = labels
        feat_importance = feat_importance.set_index('index')
        feat_importance = feat_importance.sort_values(['values'], ascending=False)
        print("Features importance: %s" %feat_importance)

    print("Accuracy for %s = %s" %(pred_model, round(100.*modl.score(X_test, y_test), 2)))


if __name__ == "__main__":
    in_dir = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/ConversionRate/Data/'
    out_dir_plot = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/ConversionRate/Plots/'
    file = in_dir + "conversion_data.csv"
    all_data = data_extraction(file)
    all_data = age_groups(all_data)
    investigation(all_data)
    ctry, src, agp, usr, usrctry, usrsrc, usragp = current_conversion_rates(all_data)
    agg_cntry, agg_src, aggr_agp = current_pages_visited(all_data, 'mean')
    # agg_cntry, agg_src, aggr_agp = current_pages_visited(all_data, 'median')

    model_building(all_data, 'Random Forest')
    model_building(all_data, 'Random Forest grid search')

    model_building(all_data, 'SVM Linear')
    model_building(all_data, 'SVM RBF')
    model_building(all_data, 'Logistic Regression')
    model_building(all_data, 'kNN')

""" CHECK OLS AFTER CREATING DUMMY VARIABLES FOR THE TEXT VARIABLES """


# Results:
# Dimensions: %s (316198, 6)
# Summary:                  age       new_user  total_pages_visited      converted
# count  316198.000000  316198.000000        316198.000000  316198.000000
# mean       30.569311       0.685469             4.872918       0.032252
# std         8.268958       0.464329             3.341053       0.176669
# min        17.000000       0.000000             1.000000       0.000000
# 25%        24.000000       0.000000             2.000000       0.000000
# 50%        30.000000       1.000000             4.000000       0.000000
# 75%        36.000000       1.000000             7.000000       0.000000
# max        79.000000       1.000000            29.000000       1.000000
# Correlation matrix:
#                           age  new_user  total_pages_visited  converted
# age                  1.000000  0.012445            -0.046093  -0.089199
# new_user             0.012445  1.000000            -0.082522  -0.152338
# total_pages_visited -0.046093 -0.082522             1.000000   0.528975
# converted           -0.089199 -0.152338             0.528975   1.000000
# Country distribution:
#  country
# China       76602
# Germany     13055
# UK          48449
# US         178092
# Name: age, dtype: int64
# Source distribution:
#  source
# Ads        88739
# Direct     72420
# Seo       155039
# Name: age, dtype: int64
