import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

plt.ion()


def data_extraction(file):

    dframe = pd.read_csv(file)
    print "Dimensions: ", dframe.shape
    print("Summary: %s" %dframe.describe())
    print dframe.head()
    dframe = dframe[dframe['seniority'] < 40.]  # removes the 2 outliers
    print "Dimensions after outliers removal: ", dframe.shape

    return dframe


def diagnostics(df0):
    dframe = df0.copy()
    for col in dframe.columns:
        if col not in ['dept', 'join_date', 'quit_date']:
            print("Non empty rows for %s: %s" %(col, np.sum(np.isfinite(dframe[col].ravel()))))
        else:
            subset = dframe[dframe[col] != '']
            print "Non empty rows for ", col, ':', subset.shape[0]
    # No data are missing


    # Uniqueness of employee #:
    print("There are %s unique employee IDs" %len(list(set(dframe['employee_id']))))


def headcount(df0):
    dframe = df0.copy()
    dframe['join_date'] = [datetime.strptime(x, "%Y-%m-%d") for x in dframe['join_date']]
    dframe['quit_date'] = [datetime.strptime(x, "%Y-%m-%d") if isinstance(x, str)
                           else np.nan for x in dframe['quit_date']]
    dframe['quit_date_modif'] = dframe['quit_date']
    dframe['quit_date_modif'] = dframe['quit_date_modif'].fillna(datetime(2050, 1, 1))
    dframe['join_date'] = pd.to_datetime(dframe['join_date'])
    dframe['quit_date_modif'] = pd.to_datetime(dframe['quit_date_modif'])
    first_day = datetime(2011,1,24)
    last_day = datetime(2015,12,13)
    days = np.arange(first_day, last_day, timedelta(days=1)).astype(datetime)

    headcount_df = pd.DataFrame([])
    filename = in_dir + 'headcounts_table.csv'
    if not os.path.exists(filename):
        for dd in days:
            print("Day %s" %dd)
            subset = dframe[(dframe['join_date'] <= dd) & (dframe['quit_date_modif'] >= dd)]
            grouped = pd.DataFrame(subset.groupby('company_id')['employee_id'].count())
            # print "grouped", grouped
            grouped['company_id'] = grouped.index
            grouped['day'] = dd
            grouped = grouped.set_index('day')

            headcount_df = headcount_df.append(grouped)

        headcount_df = headcount_df.rename(columns={"employee_id": "head_count"})
        headcount_df.to_csv(filename)
    else:
        headcount_df = pd.read_csv(filename)
        headcount_df = headcount_df.set_index('day')

    return dframe, headcount_df


def plot_headcount(df0):
    evolname = out_dir_plots + 'Headcount_by_company_over_time.png'
    if not os.path.exists(evolname):
        dframe = df0.copy()
        dframe['index'] = dframe.index
        pivoted = dframe.pivot('index', 'company_id', 'head_count')
        print "pivoted.head()", pivoted.head()
        print "pivoted.tail()", pivoted.tail()
        pivoted.index = pd.to_datetime(pivoted.index)
        pivoted.plot(figsize=(12,8))
        plt.ylabel('Head count')
        plt.title('Evolution of head count over time')
        plt.savefig(evolname)
        plt.close()


def new_features(df0):
    dframe = df0.copy()
    dframe['quit_flag'] = 0
    dframe.ix[dframe['quit_date'] < datetime(2015,12,14), 'quit_flag'] = 1
    dframe['tenure'] = dframe['quit_date_modif'] - dframe['join_date']
    dframe['tenure'] = [x.days for x in dframe['tenure']]
    # Employees with tenure > 1784 days == haven't quit yet

    return dframe


def distributions(df0, group):
    dframe = df0.copy()

    if group != 'all':
        subset = dframe[dframe['company_id'] == group]
    else:
        subset = dframe

    for col in ['seniority', 'salary', 'tenure']:
        plotname = out_dir_plots + col + '_' + str(group) + '_histogram.png'
        if not os.path.exists(plotname):
            if col == 'salary':
                rrange = range(0,450000, 5000)
            elif col == 'seniority':
                rrange = range(0,30,2)
            else:
                rrange = range(0, 15000, 300)

            plt.figure(40)
            subset[col].plot(kind='hist', bins=rrange)
            plt.xlabel(col)
            plt.savefig(plotname)
            plt.close()


def groupings(df0):
    dframe = df0.copy()
    dframe['salary_group'] = ''
    dframe.ix[dframe['salary'] < 50000.,'salary_group'] = '<50k'
    dframe.ix[(dframe['salary'] >= 50000.) & (dframe['salary'] < 150000.),'salary_group'] = '50-150k'
    dframe.ix[(dframe['salary'] >= 150000.) & (dframe['salary'] < 250000.),'salary_group'] = '150-250k'
    dframe.ix[dframe['salary'] >= 250000,'salary_group'] = '>=250k'

    dframe['tenure_group'] = ''
    dframe.ix[dframe['tenure'] < 365.,'tenure_group'] = '<1y'
    dframe.ix[(dframe['tenure'] >= 365.) & (dframe['tenure'] < 730.),'tenure_group'] = '1-2y'
    dframe.ix[(dframe['tenure'] >= 730.) & (dframe['tenure'] < 1095.),'tenure_group'] = '2-3y'
    dframe.ix[(dframe['tenure'] >= 1095.) & (dframe['tenure'] < 1460.),'tenure_group'] = '3-4y'
    dframe.ix[(dframe['tenure'] >= 1460.) & (dframe['tenure'] < 1825.),'tenure_group'] = '4-5y'
    dframe.ix[dframe['tenure'] >= 1825,'tenure_group'] = '>=5y'

    dframe['seniority_group'] = ''
    dframe.ix[dframe['seniority'] < 5.,'seniority_group'] = '<5y'
    dframe.ix[(dframe['seniority'] >= 5.) & (dframe['seniority'] < 10.),'seniority_group'] = '5-10y'
    dframe.ix[(dframe['seniority'] >= 10.) & (dframe['seniority'] < 20.),'seniority_group'] = '10-20y'
    dframe.ix[dframe['seniority'] >= 20,'seniority_group'] = '>=20y'

    # for field in ['salary_group', 'tenure_group', 'seniority_group']:
    for field in ['salary_group', 'seniority_group', 'dept']:
        group = pd.DataFrame(dframe.groupby(field)['quit_flag'].mean())
        group = group.sort_values(['quit_flag'], ascending=False)
        groupname = out_dir_plots + 'Average_quit_rate_' + field + '.png'
        if not os.path.exists(groupname):
            group.plot(kind='bar')
            plt.ylabel('Perc. quit')
            plt.savefig(groupname)
            plt.close()


def correl(df0):
    dframe = df0.copy()
    dframe = dframe.drop(['employee_id', 'company_id'],axis=1)
    print dframe.head()
    corr_matrix = dframe.corr()
    print("Correlation matrix:")
    print(corr_matrix)

    return corr_matrix


def data_preparation(df0):
    dframe = df0.copy()
    dframe = dframe.drop(['employee_id', 'join_date', 'quit_date', 'quit_date_modif'], axis=1)

    # Normalizes the numeric values that are not binary
    list_to_normal = ['seniority', 'tenure', 'salary']
    for col in list_to_normal:
        dframe[col] = preprocessing.scale(dframe[col])

    # Creates pseudo-dummy variables for categorical data
    dframe_extended = pd.get_dummies(dframe, prefix=['dept'])

    return dframe_extended


def train_test(df0):
    dframe = df0.copy()
    X = dframe.drop(['quit_flag', 'tenure'], axis=1).as_matrix()
    y = np.asarray(dframe['quit_flag'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)

    return X_train, X_test, y_train, y_test


def model_building(df0, pred_model):

    dframe = df0.copy()
    dframe = data_preparation(dframe)

    X_train, X_test, y_train, y_test = train_test(dframe)

    print("Building model %s" %pred_model)
    if pred_model == 'Random Forest':
        modl = RandomForestClassifier()
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
    y_pred = modl.predict(X_test)

    if pred_model == 'Random Forest':
        labels = list(dframe.columns.values)
        labels.remove('quit_flag')
        labels.remove('tenure')
        feat_importance = pd.DataFrame(modl.feature_importances_, columns=['values'])
        feat_importance['index'] = labels
        feat_importance = feat_importance.set_index('index')
        feat_importance = feat_importance.sort_values(['values'], ascending=False)
        print("Features importance: %s" %feat_importance)

    print("Accuracy for %s = %s" %(pred_model, round(100.*modl.score(X_test, y_test), 2)))
    print confusion_matrix(y_test, y_pred)



if __name__ == '__main__':
    in_dir = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/EmployeeRetention/Data/'
    out_dir_plots = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/EmployeeRetention/Plots/'
    file = in_dir + 'employee_retention_data.csv'
    df = data_extraction(file)
    diagnostics(df)
    df, head_df = headcount(df)
    plot_headcount(head_df)
    df_new = new_features(df)
    distributions(df_new, 'all')
    for comp_id in list(set(df_new['company_id'])):
        distributions(df_new, comp_id)
    groupings(df_new)
    corr_matrix = correl(df_new)
    dframe_extended = data_preparation(df_new)
    model_building(df_new, 'Random Forest')