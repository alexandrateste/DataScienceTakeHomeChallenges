import pandas as pd
import numpy as np
from numpy import diff
import os
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

plt.ion()


def data_extraction(file, categorical_var_list):
    dframe = pd.read_csv(file)
    print("Dimensions: ", dframe.shape))
    print("Summary: %s" %dframe.describe())
    for col in dframe.columns:
        if col not in categorical_var_list:
            print("Non empty rows for %s: %s" %(col, np.sum(np.isfinite(dframe[col].ravel()))))
        else:
            subset = dframe[dframe[col] != '']
            print("Non empty rows for ", col, ':', subset.shape[0])
    # No data are missing

    return dframe


def data_compilation(all_data, emails, clicks):
    dframe = all_data.copy()

    dframe['opened_email'] = [1 if x in emails['email_id'].values else 0 for x in dframe['email_id']]
    dframe['clicked_link'] = [1 if x in clicks['email_id'].values else 0 for x in dframe['email_id']]

    return dframe


def plotting(df0):
    dframe = df0.copy()
    # Percentage opened emails plots:
    ranges = dict(hour=np.arange(0, 25, 1),
                  user_past_purchases=np.arange(0, dframe['user_past_purchases'].max(), 1),
                  opened_email='',
                  clicked_link='')
    units = dict(hour='Local time',
                  user_past_purchases='Count',
                  opened_email='Opened email (1) or not (0)',
                  clicked_link='Click link (1) or not (0)')

    for col in ranges.keys():
        if not os.path.exists(out_dir_plot + col + '_distribution.png'):
            plt.figure(10)
            if ranges[col] == '':
                dframe[col].hist(weights=100.*np.ones_like(dframe[dframe.columns[0]] * 100.) / len(dframe))
            else:
                dframe[col].hist(bins=ranges[col], weights=100.*np.ones_like(dframe[dframe.columns[0]] * 100.) / len(dframe))
            plt.ylabel('Frequency (%)')
            plt.xlabel(units[col])
            plt.title(col)
            plt.tight_layout()
            plt.savefig(out_dir_plot + col + '_distribution.png')
            plt.close()

    # From: http://stackoverflow.com/questions/17874063/is-there-a-parameter-in-matplotlib-pandas-to-have-the-y-axis-of-a-histogram-as-p#

    for col in non_num_columns:
        if not os.path.exists(out_dir_plot + col + '_distribution.png'):
            plt.figure(10)
            dframe[col].value_counts().plot(kind='bar')
            plt.ylabel('Counts')
            plt.title(col)
            plt.tight_layout()
            plt.savefig(out_dir_plot + col + '_distribution.png')
            plt.close()


def purchase_groups(df0):
    dframe = df0.copy()
    dframe['purchase_group'] = ''
    dframe.ix[dframe['user_past_purchases'] < 5.,'purchase_group'] = '<5'
    dframe.ix[(dframe['user_past_purchases'] >= 5.) & (dframe['user_past_purchases'] < 10.),'purchase_group'] = '5-9'
    dframe.ix[(dframe['user_past_purchases'] >= 10.) & (dframe['user_past_purchases'] < 15.),'purchase_group'] = '10-14'
    dframe.ix[dframe['user_past_purchases'] >= 15.,'purchase_group'] = '>=15'

    return dframe


def plot_by_feature(df0, metric, list_of_columns):
    dframe = df0.copy()

    for col in list_of_columns:
        if not os.path.exists(out_dir_plot + metric + '_per_' + col + '.png'):
            plt.figure(10)
            grouped = pd.DataFrame(100. * dframe.groupby(col)[metric].sum() / dframe.groupby(col)[metric].count())
            grouped = grouped.sort_values([metric], ascending=False)
            grouped.plot(kind='bar')
            plt.ylabel('Percentage ' + metric + ' (%)')
            plt.title(metric + ' per ' + col)
            plt.tight_layout()
            plt.savefig(out_dir_plot + metric + '_per_' + col + '.png')
            plt.close()


def day_hour_heatmap(df0, metric, email_type):
    dframe = df0.copy()
    if email_type:
        subset = dframe[dframe[email_type[0]] == email_type[1]]
    else:
        subset = dframe.copy()

    # Percentage wrt all emails sent out
    # grouped = 100. * subset.groupby(['weekday', 'hour'])[metric].count().unstack() / subset.shape[0]

    # Percentage wrt all emails sent out on a given day
    grouped = subset.groupby(['weekday', 'hour'])[metric].count().unstack()
    grouped = grouped.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    # At this point, rows = days (sorted as in the week), columns = hours, and values = % wrt overall emails count

    grouped_perc = grouped.apply(lambda x: x / x.sum() * 100, axis=1)
    grouped_perc = grouped_perc.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Conversion of the data frames into a matrices for heatmap representation
    matr = grouped_perc.as_matrix().transpose()

    if not os.path.exists(out_dir_plot + metric + '_' + email_type[1] + '_per_weekday_hour_heatmap.png'):
        fig = plt.figure(10)
        ax = fig.add_subplot(111)
        im = ax.imshow(matr, cmap='hot_r', interpolation='nearest')
        ax.set_xlabel('xlabel')
        ax.set_aspect(0.35)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Percentage (%)')
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
        ax.set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
        ax.set_ylabel('Hours')
        ax.set_xlabel('')
        fig.suptitle('Weekday vs. hour ' + metric + ' heatmap for ' + email_type[1], fontsize=15)
        fig.savefig(out_dir_plot + metric + '_' + email_type[1] + '_per_weekday_hour_heatmap.png')
        plt.close()


def statistical_results(df0):
    dframe = df0.copy()

    print("--------------")
    print("    Overall   ")
    print("--------------")
    print("%s %% of emails were opened" %round(100.*dframe['opened_email'].sum()/dframe.shape[0], 2))
    print("%s %% of links were clicked" %round(100.*dframe['clicked_link'].sum()/dframe.shape[0], 2))

    subset = dframe[dframe['opened_email'] == 1]
    print("%s %% of opened emails had their links clicked"
          %(100.*subset['clicked_link'].sum()/subset.shape[0]))

    for country in list(set(dframe['user_country'])):
        print("--------------")
        print("    In %s   " %country)
        print("--------------")
        dframe_sub = dframe[dframe['user_country'] == country]
        print("%s %% of emails were opened"
              %round(100.*dframe_sub['opened_email'].sum()/dframe_sub.shape[0],2))
        print("%s %% of links were clicked"
              %round(100.*dframe_sub['clicked_link'].sum()/dframe_sub.shape[0],2))

        subset = dframe_sub[dframe_sub['opened_email'] == 1]
        print("%s %% of opened emails had their links clicked"
              %round(100.*subset['clicked_link'].sum()/subset.shape[0],2))


def correl(df0):
    dframe = df0.copy()
    dframe = dframe.drop(['email_id', 'purchase_group'],axis=1)
    print(dframe.head())
    corr_matrix = dframe.corr()
    print("Correlation matrix:")
    print(corr_matrix)

    return corr_matrix


def train_test(df0, outcome_variable):
    dframe = df0.copy()
    X = dframe.drop([outcome_variable], axis=1).as_matrix()
    y = np.asarray(dframe[outcome_variable])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)

    return X_train, X_test, y_train, y_test


def data_preparation(df_train, df_test, with_without):
    dftrain = df_train.copy()
    dftest = df_test.copy()
    if with_without == 'with':
        list_to_remove = ['email_id', 'purchase_group']
        # list_to_dummify = ['opened_email', 'weekday', 'user_country', 'email_text', 'email_version']
    else:
        list_to_remove = ['email_id', 'purchase_group', 'opened_email', 'email_version', 'email_text']
        # list_to_dummify = ['weekday', 'user_country']

    dftrain = dftrain.drop(list_to_remove, axis=1)
    dftest = dftest.drop(list_to_remove, axis=1)
    # Removing 'opened_email' because we cannot know if a user will open the email before we send it
    # print dftrain.columns

    # Normalizes the numeric values that are not binary
    list_to_normal = ['hour', 'user_past_purchases']
    scaler = preprocessing.StandardScaler()
    for col in list_to_normal:
        dftrain[col] = scaler.fit_transform(dftrain[col])
        dftest[col] = scaler.transform(dftest[col])
        # Help from: http://scikit-learn.org/stable/modules/preprocessing.html

    # Creates pseudo-dummy variables for categorical data
    dftrain_extended = pd.get_dummies(dftrain)  #, prefix=list_to_dummify <-- ends up not being in the proper order
    dftest_extended = pd.get_dummies(dftest)

    headers = list(dftrain_extended.columns)

    X_train_extended = dftrain_extended.as_matrix()
    X_test_extended = dftest_extended.as_matrix()

    return X_train_extended, X_test_extended, headers


def model_building(Xtrain, Xtest, ytrain, ytest, pred_model_list):

    print("-------------")
    print("Probabilities")
    print("-------------")
    plt.figure(200)
    for pred_model in pred_model_list:
        print("Now running %s" %pred_model)
        if pred_model == 'Random Forest':
            modl = RandomForestClassifier()
        elif pred_model == 'SVM Linear':
            modl = SVC(kernel='linear', probability=True)
        elif pred_model == 'SVM RBF':
            modl = SVC(kernel='rbf', probability=True)
        elif pred_model == 'Logistic Regression':
            modl = LogisticRegression()

        # ROC curve - search for best threshold
        probas_ = modl.fit(Xtrain, ytrain).predict_proba(Xtest)
        fpr, tpr, thresholds = roc_curve(ytest, probas_[:, 1])
        plt.plot(fpr, tpr, marker='o', markersize=2, markeredgecolor='None', label=pred_model)

    plt.plot([0,1],[0,1], linestyle='--', color='grey')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend(loc='best')
    plt.savefig(out_dir_plot + 'ROC_curve_allfeatures_'+wwt+'_openedemail.png')
    plt.close()


def model_building_prediction(Xtrain, Xtest, ytrain, ytest, pred_model_list, headers):

    # Usual prediction (threshold = 0.5)
    print("----------------")
    print("Usual prediction")
    print("----------------")
    for pred_model in pred_model_list:
        print("Now running %s" %pred_model)
        if pred_model == 'Random Forest':
            modl = RandomForestClassifier()
        elif pred_model == 'SVM Linear':
            modl = SVC(kernel='linear')
        elif pred_model == 'SVM RBF':
            modl = SVC(kernel='rbf')
        elif pred_model == 'Logistic Regression':
            modl = LogisticRegression()

        modl.fit(Xtrain, ytrain)
        y_pred = modl.predict(Xtest)
        conf_matrix = confusion_matrix(ytest, y_pred)
        confusion_df = pd.DataFrame(conf_matrix, columns=['Pred_clicked_link_0', 'Pred_clicked_link_1'])
        confusion_df.index = ['True_clicked_link_0', 'True_clicked_link_1']
        # Help for headers and indices from: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

        if pred_model == 'Random Forest':
            labels = headers
            feat_importance = pd.DataFrame(modl.feature_importances_, columns=['values'])
            feat_importance['index'] = labels
            feat_importance = feat_importance.set_index('index')
            feat_importance = feat_importance.sort_values(['values'], ascending=False)
            print("Features importance: %s" % feat_importance)

        print("Confusion matrix:\n %s" % confusion_df)

        print("Accuracy for %s = %s %%" % (pred_model, round(100. * modl.score(Xtest, ytest), 2)))
        print("Sensitivity (i.e. Recall) for %s = %s %%" % (pred_model,
                                                            round(100. * conf_matrix[1, 1] / (conf_matrix[1, 1]
                                                                                              + conf_matrix[1, 0]),
                                                                  2)))
        if conf_matrix[1, 1] + conf_matrix[0, 1]:
            print("Precision for %s = %s %%" % (pred_model, round(100. * conf_matrix[1, 1] /
                                                                  (conf_matrix[1, 1] + conf_matrix[0, 1]), 2)))


def best_model(Xtrain, Xtest, ytrain, ytest, pred_model):

    # Usual prediction (threshold = 0.5)
    print("Best model from ROC curve: %s" %pred_model)
    if pred_model == 'Random Forest':
        modl = RandomForestClassifier()
    elif pred_model == 'SVM Linear':
        modl = SVC(kernel='linear', probability=True)
    elif pred_model == 'SVM RBF':
        modl = SVC(kernel='rbf', probability=True)
    elif pred_model == 'Logistic Regression':
        modl = LogisticRegression()

    # ROC curve - search for best threshold
    probas_ = modl.fit(Xtrain, ytrain).predict_proba(Xtest)
    fpr, tpr, thresholds = roc_curve(ytest, probas_[:, 1])

    return fpr, tpr, thresholds


def pick_best_threshold(fpr, thresholds):
    index = np.argmax(diff(fpr))
    best_threshold = thresholds[index]

    return best_threshold


def optimized_model_initial(Xtrain, Xtest, ytrain, ytest, thresh):
    # Prediction with best threshold
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("Optimized prediction -- with ROC curve threshold = %s" %round(1-thresh,2))
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    modl = LogisticRegression(class_weight={0: 1-thresh, 1: thresh})

    modl.fit(Xtrain, ytrain)
    y_pred = modl.predict(Xtest)

    conf_matrix = confusion_matrix(ytest, y_pred)
    confusion_df = pd.DataFrame(conf_matrix, columns=['Pred_clicked_link_0', 'Pred_clicked_link_1'])
    confusion_df.index = ['True_clicked_link_0', 'True_clicked_link_1']
    # Help for headers and indices from: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    print("Confusion matrix:\n %s" % confusion_df)

    print("Accuracy = %s %%" % round(100. * modl.score(Xtest, ytest), 2))
    print("Sensitivity (i.e. Recall) = %s %%" % round(100. * conf_matrix[1, 1] /
                                                      (conf_matrix[1, 1] + conf_matrix[1, 0]), 2))
    if conf_matrix[1, 1] + conf_matrix[0, 1]:
        print("Precision = %s %%" % round(100. * conf_matrix[1, 1] /
                                                 (conf_matrix[1, 1] + conf_matrix[0, 1]), 2))


def optimized_model(Xtrain, Xtest, ytrain, ytest):
    # Prediction with best threshold
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print("Optimized prediction -- Searching for best threshold")
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - ")
    current_sensitivity = 0.
    best_thresh = 0.
    for thresh in np.arange(0,1,0.01):
        # print("\n>>> Threshold = %s <<<" %thresh)
        modl = LogisticRegression(class_weight={0: 1-thresh, 1: thresh})

        modl.fit(Xtrain, ytrain)
        y_pred = modl.predict(Xtest)

        conf_matrix = confusion_matrix(ytest, y_pred)
        confusion_df = pd.DataFrame(conf_matrix, columns=['Pred_clicked_link_0', 'Pred_clicked_link_1'])
        confusion_df.index = ['True_clicked_link_0', 'True_clicked_link_1']
        # Help for headers and indices from: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

        # print("Confusion matrix:\n %s" % confusion_df)
        # print("Accuracy = %s %%" % round(100. * modl.score(Xtest, ytest), 2))

        sensitivity = 100. * conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
        # print("Sensitivity (i.e. Recall) = %s %%" % round(sensitivity, 9))

        if conf_matrix[1, 1] + conf_matrix[0, 1]:
            precision = 100. * conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
            # print("Precision for = %s %%" % round(precision, 2))

        if abs(sensitivity - current_sensitivity) > 1e-6:
            current_sensitivity = sensitivity
            best_thresh = thresh

    print("Best sensitivity (%s %%) obtained for a threshold of %s for class 0"
          %(round(sensitivity,2), round(1-best_thresh,2)))

    return sensitivity, best_thresh


def double_grid_search(Xtrain, Xtest, ytrain, ytest):
    # Prediction with best threshold
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print("Optimized prediction -- Searching for best threshold")
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - ")
    sensitivity_list = []
    precision_list = []
    for weight in np.arange(0,1,0.01):
        print("\n>>> Weight = %s <<<" % weight)
        sens_list = []
        prec_list = []
        for threshold in np.arange(0,1,0.1):
            print("\n>>> Threshold = %s <<<" %threshold)
            modl = LogisticRegression(class_weight={0: 1-weight, 1: weight})

            probas_matr = modl.fit(Xtrain, ytrain).predict_proba(Xtest)
            y_pred = [1 if x > threshold else 0 for x in probas_matr[:, 1]]

            conf_matrix = confusion_matrix(ytest, y_pred)
            confusion_df = pd.DataFrame(conf_matrix, columns=['Pred_clicked_link_0', 'Pred_clicked_link_1'])
            confusion_df.index = ['True_clicked_link_0', 'True_clicked_link_1']
            # Help for headers and indices from: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

            print("Confusion matrix for highest precision:\n %s" % confusion_df)
            # print("Accuracy = %s %%" % round(100. * modl.score(Xtest, ytest), 2))

            sensitivity = 100. * conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
            # print("Sensitivity (i.e. Recall) = %s %%" % round(sensitivity, 9))

            if conf_matrix[1, 1] + conf_matrix[0, 1]:
                precision = 100. * conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
                prec_list.append(precision)
                # print("Precision for = %s %%" % round(precision, 2))
            sens_list.append(sensitivity)
        sensitivity_list.append(sens_list)
        precision_list.append(prec_list)

    return sensitivity_list, precision_list


def sens_prec_heatmap(sensitivity_list, precision_list):
    precision_list2 = [x if len(x) == 10 else list(np.ones(10) * np.nan) for x in precision_list]
    sensitivity_list2 = [x if len(x) == 10 else list(np.ones(10) * np.nan) for x in sensitivity_list]
    heatmap_plot('precision', precision_list2, 200)
    heatmap_plot('sensitivity', sensitivity_list2, 100)


def heatmap_plot(list_type, list_of_data, plot_index):
    fig = plt.figure(plot_index)
    ax = fig.add_subplot(111)
    im = ax.imshow(list_of_data, cmap='hot_r', interpolation='nearest')
    ax.set_aspect(0.1)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Percentage (%)')
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_ylabel('Weight (%)')
    ax.set_xlabel('Threshold (%)')
    fig.suptitle(list_type + ' vs. weight and threshold heatmap', fontsize=15)
    fig.savefig(out_dir_plot + list_type + '_weight_threshold_heatmap.png')
    plt.close()


def sens_prec_determination(sensitivity_list, precision_list):
    precision_list2 = [x if len(x) == 10 else list(np.ones(10) * np.nan) for x in precision_list]
    sensitivity_list2 = [x if len(x) == 10 else list(np.ones(10) * np.nan) for x in sensitivity_list]
    sensitivity_list3 = [[x if x != 100.0 else -999. for x in element] for element in sensitivity_list2]
    sensitivity_list4 = [[x if x != 100.0 else np.nan for x in element] for element in sensitivity_list2]
    precision_list3 = [[x if x != 100.0 else -999. for x in element] for element in precision_list2]
    precision_list4 = [[x if x != 100.0 else np.nan for x in element] for element in precision_list2]
    sensmatr = np.array(sensitivity_list3)
    precmatr = np.array(precision_list3)
    precmatr4plot = np.array(precision_list4)
    sensmatr4plot = np.array(sensitivity_list4)

    highest_precision = []
    for kk in range(0, 10):
        highest_precision.append(precmatr[sensmatr[:, kk].argmax(), kk])

    maxprec = round(np.nanmax(highest_precision), 2)
    maxthresh = np.nanargmax(highest_precision)
    maxweight = sensmatr[:, maxthresh].argmax()

    print("Maximum precision of %s %% for small class weight=%s %% and threshold=%s %%" % (maxprec, maxweight, maxthresh*10.))

    colors = ['blue', 'green', 'red', 'cyan', 'purple', 'olive', 'black', 'pink', 'orange', 'gold']

    plt.figure(99)
    for k in range(0, 10):
        plt.plot(sensmatr4plot[:, k], marker='o', color=colors[k], label=str(k*10) + ' %')
    plt.ylabel('Sensitivity (%)')
    plt.xlabel('Big class weight (%)')
    plt.legend(loc='best')
    plt.savefig(out_dir_plot + 'Sensitivity_vs_weight_threshold_lines.png')
    plt.close()

    plt.figure(98)
    for k in range(0, 10):
        plt.plot(precmatr4plot[:, k], marker='o', color=colors[k], label=str(k*10) + ' %')
    plt.ylabel('Precision (%)')
    plt.xlabel('Big class weight (%)')
    plt.legend(loc='best')
    plt.savefig(out_dir_plot + 'Precision_vs_weight_threshold_lines.png')
    plt.close()


if __name__ == "__main__":
    in_dir = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/MarketingEmailCampaign/Data/'
    out_dir_plot = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/MarketingEmailCampaign/Plots/'
    all_data_file = in_dir + 'email_table.csv'
    opened_file = in_dir + 'email_opened_table.csv'
    clicked_file = in_dir + 'link_clicked_table.csv'

    non_num_columns = ['email_text', 'email_version', 'weekday', 'user_country']
    all_data = data_extraction(all_data_file, non_num_columns)
    opened_emails = data_extraction(opened_file, [])
    clicked_emails = data_extraction(clicked_file, [])

    all_data_augmented = data_compilation(all_data, opened_emails, clicked_emails)
    plotting(all_data_augmented)
    with_new_feat = purchase_groups(all_data_augmented)
    plot_by_feature(with_new_feat, 'opened_email', ['hour', 'weekday', 'user_country', 'purchase_group'])
    plot_by_feature(with_new_feat, 'clicked_link', ['email_text', 'email_version', 'hour', 'weekday',
                                                    'user_country', 'purchase_group'])
    day_hour_heatmap(with_new_feat, 'clicked_link', ['email_version', 'personalized'])
    day_hour_heatmap(with_new_feat, 'clicked_link', ['email_version', 'generic'])
    day_hour_heatmap(with_new_feat, 'clicked_link', ['email_text', 'long_email'])
    day_hour_heatmap(with_new_feat, 'clicked_link', ['email_text', 'short_email'])

    # Correlation:
    corr_matrix = correl(with_new_feat)

    # Model:
    X_train, X_test, y_train, y_test = train_test(with_new_feat, 'clicked_link')
    short_list_of_columns = list(with_new_feat.columns)
    short_list_of_columns.remove('clicked_link')
    df_train = pd.DataFrame(X_train, columns=short_list_of_columns)
    df_test = pd.DataFrame(X_test, columns=short_list_of_columns)

    for wwt in ['without', 'with']:
        print("\n********************")
        print("%s opened email and email content" %wwt)
        print("********************")
        X_train_extended, X_test_extended, headers = data_preparation(df_train, df_test, wwt)

        # pred_model_list = ['Random Forest', 'SVM Linear', 'SVM RBF', 'Logistic Regression']
        # model_building(X_train_extended, X_test_extended, y_train, y_test, pred_model_list)
        # model_building_prediction(X_train_extended, X_test_extended, y_train, y_test, pred_model_list, headers)
        if wwt == 'with':
            fpr, tpr, thresholds = best_model(X_train_extended, X_test_extended, y_train, y_test, 'Logistic Regression')
            best_threshold = pick_best_threshold(fpr, thresholds)
            # !!! the ROC curve gives the threshold for class 0 to use in LogisticRegression()
            print("\n From ROC curve")
            optimized_model_initial(X_train_extended, X_test_extended, y_train, y_test, 1-best_threshold)
            print("\n From pseudo-grid search")
            sensitivity, best_thresh = optimized_model(X_train_extended, X_test_extended, y_train, y_test)
            optimized_model_initial(X_train_extended, X_test_extended, y_train, y_test, best_thresh)

            # The method with the ROC curve is best because really give the appropriate threshold value,
            # while a loop is constrained by the step size used in the loop
            sensitivity_list, precision_list = double_grid_search(X_train_extended, X_test_extended, y_train, y_test)
            sens_prec_heatmap(sensitivity_list, precision_list)

            sens_prec_determination(sensitivity_list, precision_list)
