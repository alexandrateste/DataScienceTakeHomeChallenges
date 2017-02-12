import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO
import pydot_ng as pydot


def data_extraction(file):

    print("===========================")
    print("Extracting file %s" %file)
    print("===========================")
    dframe = pd.read_csv(file)
    print "Dimensions: ", dframe.shape
    print("Summary: %s" %dframe.describe())
    print dframe.head()

    return dframe


def diagnostics(df0, collist):
    dframe = df0.copy()
    for col in dframe.columns:
        print col
        if col not in collist:
            print("Non empty rows for %s: %s" %(col, np.sum(np.isfinite(dframe[col].ravel()))))
            histogram_plot(df0, col)
        else:
            subset = dframe[dframe[col] != '']
            print "Non empty rows for ", col, ':', subset.shape[0]
    # No data are missing


def find_country(val, country):
    print("IP address: %s" %val)
    subset = country[(country['lower_bound_ip_address'] <= val) & (country['upper_bound_ip_address'] >= val)]
    # print subset
    if subset.shape[0]:
        land = subset.values[0][2]  # extracts the country name
    else:
        land = ''

    return land


def combination(act, ctry):
    # print("Before: act.head = %s" %act.head())
    filename = in_dir + 'Combined_data.csv'
    if not os.path.exists(filename):
        act['country'] = act['ip_address'].apply(lambda x: find_country(x, ctry))
        act.to_csv(filename)

    else:
        act = pd.read_csv(filename)
        act = act.drop(['Unnamed: 0'], axis=1)

    # print("After: act.head = %s" %act.head())

    return act


def timeseries_plots(df0):
    dframe = df0.copy()
    dframe = dframe.set_index('purchase_time')
    dframe.index = pd.to_datetime(dframe.index)

    dframe['Day'] = [x.date() for x in dframe.index]

    purchase = pd.DataFrame(dframe.groupby('Day')['purchase_value'].mean())
    fraud = pd.DataFrame(dframe.groupby('Day')['class'].mean())
    purchase.index = pd.to_datetime(purchase.index)
    fraud.index = pd.to_datetime(fraud.index)
    new_user_count = pd.DataFrame(dframe.groupby('Day')['user_id'].count())
    new_user_count.index = pd.to_datetime(new_user_count.index)

    filename = out_dir_plots + 'Mean_purchase_value_timeseries.png'
    if not os.path.exists(filename):
        purchase.plot()
        plt.ylabel('Avg. purchase value (USD)')
        plt.title("Mean purchase value over time")  # Check presence of seasonalities
        plt.savefig(filename)
        plt.close()

    filename = out_dir_plots + 'Mean_fraud_percentage_timeseries.png'
    if not os.path.exists(filename):
        fraud.plot()
        plt.ylabel('Perc. fraud')
        plt.title("Mean percentage of fraud")  # Check presence of seasonalities
        plt.savefig(filename)
        plt.close()

    filename = out_dir_plots + 'New_users_count_timeseries.png'
    if not os.path.exists(filename):
        new_user_count.plot()
        plt.ylabel('Count of new users')
        plt.title("Count of new users")  # Check presence of seasonalities
        plt.savefig(filename)
        plt.close()


def proportions(df0, group):
    dframe = df0.copy()

    for gp in group:
        filename = out_dir_plots + 'Fraudsters_proportion_by_' +gp + '.png'
        if not os.path.exists(filename):
            grouped = pd.DataFrame(dframe.groupby(gp)['class'].mean())
            grouped = grouped.sort_values(['class'], ascending=False)
            if gp == 'country':
                grouped.plot(kind='bar', legend=False, figsize=(30,8))
            else:
                grouped.plot(kind='bar', legend=False)
            plt.ylabel('Perc. fraudsters')
            plt.title("Proprotion of fraudsters amoung new users by " + gp)  # Check presence of seasonalities
            plt.savefig(filename)
            plt.close()


def new_features(df0):
    dframe = df0.copy()
    dframe['signup_time'] = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dframe['signup_time']]
    dframe['purchase_time'] = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dframe['purchase_time']]
    dframe['time_to_purchase'] = dframe['purchase_time'] - dframe['signup_time']
    dframe['time_to_purchase'] = [x.total_seconds()/86400.for x in dframe['time_to_purchase']]

    dframe['age_group'] = ''
    dframe.ix[dframe['age'] < 25.,'age_group'] = '<25'
    dframe.ix[(dframe['age'] >= 25.) & (dframe['age'] < 35.),'age_group'] = '25-34'
    dframe.ix[(dframe['age'] >= 35.) & (dframe['age'] < 45.),'age_group'] = '35-44'
    dframe.ix[dframe['age'] >= 45.,'age_group'] = '>=45'

    dframe['time_to_purchase_group'] = ''
    dframe.ix[dframe['time_to_purchase'] < 30., 'time_to_purchase_group'] = '<1month'
    dframe.ix[(dframe['time_to_purchase'] >= 30.) & (dframe['time_to_purchase'] < 60.), 'time_to_purchase_group'] = '1-2months'
    dframe.ix[(dframe['time_to_purchase'] >= 60.) & (dframe['time_to_purchase'] < 90.), 'time_to_purchase_group'] = '2-3months'
    dframe.ix[dframe['time_to_purchase'] >= 90., 'time_to_purchase_group'] = '>=3months'

    dframe['purchase_value_group'] = ''
    dframe.ix[dframe['purchase_value'] < 20., 'purchase_value_group'] = '<$20'
    dframe.ix[(dframe['purchase_value'] >= 20.) & (dframe['purchase_value'] < 40.), 'purchase_value_group'] = '$20-40'
    dframe.ix[(dframe['purchase_value'] >= 40.) & (dframe['purchase_value'] < 60.), 'purchase_value_group'] = '$40-60'
    dframe.ix[dframe['purchase_value'] >= 60., 'purchase_value_group'] = '>=$60'

    return dframe


def histogram_plot(df0, col, zoom=None):
    dframe = df0.copy()
    dframe[col].plot(kind='hist', bins=100)
    if zoom:
        plt.ylim(zoom)
        suffix = 'ZoomIn'
    else:
        suffix = ''
    plt.xlabel(col)
    plt.title("Distribution of " + col)
    plt.savefig(out_dir_plots + col + '_distribution' + suffix + '.png')
    plt.close()


def correl(df0, collist):
    dframe = df0.copy()
    dframe = dframe.drop(collist,axis=1)
    # print dframe.head()
    corr_matrix = dframe.corr()
    print("Correlation matrix:")
    print(corr_matrix)

    return corr_matrix


def data_preparation(df0):
    dframe = df0.copy()
    dframe = dframe[['class', 'purchase_value', 'age', 'time_to_purchase', 'source', 'browser', 'sex', 'country']]

    # Normalizes the numeric values that are not binary
    list_to_normal = ['purchase_value', 'age', 'time_to_purchase']
    for col in list_to_normal:
        dframe[col] = preprocessing.scale(dframe[col])

    # Creates pseudo-dummy variables for categorical data
    dframe_extended = pd.get_dummies(dframe, prefix=['source', 'browser', 'sex', 'country'])

    return dframe_extended


def train_test(df0):
    dframe = df0.copy()
    X = dframe.drop(['class'],axis=1).as_matrix()
    y = np.asarray(dframe['class'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)

    return X_train, X_test, y_train, y_test


def fraud_prediction(df0, pred_model):
    dframe = df0.copy()
    X_train, X_test, y_train, y_test = train_test(dframe)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Running %s model" %pred_model)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    if pred_model == 'Random Forest':
        model = RandomForestClassifier()
    elif pred_model == 'Logistic Regression':
        model = LogisticRegression()
    elif pred_model == 'SVM Linear':
        model = SVC(kernel='linear')
    elif pred_model == 'SVM RBF':
        model = SVC(kernel='rbf')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    if pred_model == 'Random Forest':
        labels = list(dframe.columns.values)
        labels.remove('class')
        feat_importance = pd.DataFrame(model.feature_importances_, columns=['values'])
        feat_importance['index'] = labels
        feat_importance = feat_importance.set_index('index')
        feat_importance = feat_importance.sort_values(['values'], ascending=False)
        feat_importance = feat_importance[feat_importance['values'] > 0.001]
        print("Features importance (truncated table): %s" % feat_importance)

        print("Single tree to see splits")
        model_dt = DecisionTreeClassifier(random_state=12345)
        model_dt.fit(X_train, y_train)

        # feature_names = labels
        # class_name = model_dt.classes_.astype(int).astype(str)
        #
        # def output_pdf(clf_, name):
        #     dot_data = StringIO()
        #     tree.export_graphviz(clf_, out_file=dot_data,
        #                          feature_names=feature_names,
        #                          class_names=class_name,
        #                          filled=True, rounded=True,
        #                          special_characters=True,
        #                          node_ids=1, )
        #     graph = pydot.graph_from_dot_data(dot_data.getvalue())
        #     graph.write_pdf("%s.pdf" % name)
        # n = 10
        # output_pdf(model_dt, name='filename%s' %n)


        model_dt.predict(X_test)
        decision_path = model_dt.decision_path(X_train, check_input=True)

        print("Decision path:\n %s" % decision_path)

    print("Confusion matrix:\n %s" %conf_matrix)

    print("Accuracy for %s = %s %%" % (pred_model, round(100. * model.score(X_test, y_test), 2)))
    print("Sensitivity (i.e. Recall) for %s = %s %%" % (pred_model,
                                                        round(100. * conf_matrix[0,0]/(conf_matrix[0,0]
                                                                                    + conf_matrix[0,1]), 2)))
    print("Precision for %s = %s %%" % (pred_model,
                                        round(100. * conf_matrix[0,0]/(conf_matrix[0,0]+
                                                                    conf_matrix[1,0]), 2)))

    return conf_matrix


if __name__ == "__main__":
    in_dir = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/FraudulentActivities/Data/Fraud/'
    out_dir_plots = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/FraudulentActivities/Plots/'
    activityfile = in_dir + 'Fraud_Data.csv'
    countryfile = in_dir + 'IpAddress_to_Country.csv'
    activity = data_extraction(activityfile)
    country = data_extraction(countryfile)
    # print("Countries list:\n")
    # print(list(set(country['country'])))

    # diagnostics(activity, ['signup_time', 'purchase_time', 'device_id', 'source', 'browser', 'sex'])
    # diagnostics(country, ['country'])

    timeseries_plots(activity)

    master_df = combination(activity, country)

    df_new = new_features(master_df)

    corr_matrix = correl(df_new, ['user_id', 'signup_time', 'purchase_time',
                                  'device_id', 'source', 'browser', 'sex',
                                  'ip_address', 'country', 'age_group',
                                  'time_to_purchase_group', 'purchase_value_group'])


    # histogram_plot(df_new, 'time_to_purchase')
    # histogram_plot(df_new, 'time_to_purchase', zoom=[0, 2000])

    # proportions(df_new, ['age_group', 'sex', 'country', 'source', 'browser',
    #                      'time_to_purchase_group', 'purchase_value_group'])

    processed_df = data_preparation(df_new)

    confus_matrix_rf = fraud_prediction(processed_df, 'Random Forest')
    confus_matrix_lr = fraud_prediction(processed_df, 'Logistic Regression')
    # confus_matrix_svmlin = fraud_prediction(processed_df, 'SVM Linear')