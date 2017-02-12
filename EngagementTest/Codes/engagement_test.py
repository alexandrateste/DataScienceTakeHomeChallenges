import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy import stats
import statsmodels.stats.api as sms

plt.ion()

def data_extraction(file):

    #Extracts data from file and save into a dataframe
    print("=======================")
    print("File %s" %file)
    print("=======================")
    dframe = pd.read_csv(file)
    print("Dimensions: ")
    print(dframe.shape)
    print("Columns: %s" %dframe.columns)
    print("Summary: %s" %dframe.describe())
    print("Head: %s" %dframe.head())
    print("Tail: %s" %dframe.tail())

    return dframe


def diagnostics(df0, collist):

    # Checks whether there are missing data or not
    dframe = df0.copy()
    if dframe.shape[0]:
        for col in dframe.columns:
            if col not in collist:
                length = np.sum(np.isfinite(dframe[col]))
            else:
                length = len([x for x in dframe[col].values if x != ''])
            print("Percentage of non-nan records for col %s: %s%%" % (col, 100.*length/dframe.shape[0]))
    else:
        print("This dataframe is empty")

    # Uniqueness of user_id
    if 'user_id' in dframe.columns:
        expected = dframe.shape[0]
        unique_counts = len(list(set(dframe['user_id'])))
        if unique_counts == expected:
            print("There are no duplicate user_ids")
        else:
            print("There are duplicate users_ids")


def merging(df1, df2, newcol):

    merged = pd.merge(left=df1, right=df2, left_on='user_id', right_on='user_id', how='left')

    length = len([x for x in merged[newcol].values if x != ''])
    print("Percentage of non-nan records for col %s: %s%%" % (newcol, 100. * length / merged.shape[0]))

    return merged


def new_feature(df0, startdate, firstdate):
    dframe = df0.copy()
    dframe[startdate] = [datetime.strptime(x, "%Y-%m-%d") for x in dframe[startdate]]
    dframe[firstdate] = [datetime.strptime(x, "%Y-%m-%d") for x in dframe[firstdate]]
    dframe[startdate] = [x.date() for x in dframe[startdate]]
    dframe[firstdate] = [x.date() for x in dframe[firstdate]]
    dframe['tenure'] = dframe[firstdate] - dframe[startdate]
    dframe['tenure'] = [x.total_seconds()/86400. for x in dframe['tenure']]
    print("Summary: %s" %dframe.describe())

    dframe['tenure_group'] = ''
    dframe.ix[dframe['tenure'] == 0., 'tenure_group'] = '0d'
    dframe.ix[(dframe['tenure'] >=1.) & (dframe['tenure'] < 31.), 'tenure_group'] = '1-30'
    dframe.ix[(dframe['tenure'] >=31.) & (dframe['tenure'] < 91.), 'tenure_group'] = '31-90'
    dframe.ix[(dframe['tenure'] >=91.) & (dframe['tenure'] < 151.), 'tenure_group'] = '91-150'
    dframe.ix[(dframe['tenure'] >=151.) & (dframe['tenure'] < 211.), 'tenure_group'] = '151-210'
    dframe.ix[dframe['tenure'] >=211., 'tenure_group'] = '>210'

    dframe['weekday'] = [x.weekday() for x in dframe['date']]

    return dframe


def simple_histograms(df0, group, zoom=None):

    dframe = df0.copy()
    for gp in group:
        if gp in ['browser', 'date']:
            filename = out_dir_plots + gp + '_stacked_bars.png'
            if not os.path.exists(filename):
                grouped = pd.DataFrame(dframe.groupby([gp, 'test'])['user_id'].count())
                print grouped.head()
                grouped = grouped.unstack()
                grouped = grouped.sort_values(['user_id'], ascending=False)
                print grouped.head()
                plt.figure(10)
                grouped.plot(kind='bar', stacked=True)
                plt.savefig(filename)
                plt.close()
        else:
            if zoom:
                suffix = '_ZoomIn'
            else:
                suffix = ''
            filename = out_dir_plots + gp + '_distributions' + suffix + '.png'
            if not os.path.exists(filename):
                control = dframe[dframe['test'] == 0]
                treatment = dframe[dframe['test'] == 1]
                plt.figure(20)
                ax = control[gp].plot(kind='hist', bins=100, alpha=0.5)
                treatment[gp].plot(kind='hist', bins=100, alpha=0.5, ax=ax)
                plt.xlabel(gp)
                plt.ylabel('Frequency')
                if zoom:
                    plt.ylim(zoom)
                plt.savefig(filename)
                plt.close()


def avg_pages(df0, bar_gp, scatter_gp):
    dframe = df0.copy()
    for gp in bar_gp:
        print("Plotting for %s" %gp)
        filename = out_dir_plots + 'Average_pages_visited_' + gp + '_bars.png'
        if not os.path.exists(filename):
            grouped = dframe.groupby([gp, 'test'])['pages_visited'].mean()
            grouped = grouped.unstack()
            grouped.plot(kind='bar')
            plt.ylabel('Average # of pages visited')
            plt.savefig(filename)
            plt.close()

    for gp in scatter_gp:
        print("Plotting for %s" %gp)
        filename = out_dir_plots + gp + '_average_pages_visited_scatter.png'
        if not os.path.exists(filename):
            plt.figure(30)
            plt.plot(dframe[gp], dframe['pages_visited'], marker='o', linestyle='None')
            plt.ylabel('# of pages visited')
            plt.xlabel('tenure (days)')
            plt.savefig(filename)
            plt.close()


def ttest_unit(control, treatment):
    tstat, pvalue = stats.ttest_ind(control, treatment, equal_var=False)
    cm = sms.CompareMeans(sms.DescrStatsW(control), sms.DescrStatsW(treatment))
    conf_interval = cm.tconfint_diff(usevar='unequal')
    print("T-statistics = %s\n" %tstat)
    print("p-value = %s\n" %pvalue)
    print("95% confidence Interval = ")
    print(conf_interval)



def t_tests(df0, group):
    dframe = df0.copy()
    dframe = dframe[np.isfinite(dframe['pages_visited'])]

    if group != 'all':
        list_of_interest = list(set(dframe[group]))
        for element in list_of_interest:
            print("\n-------------------")
            print("T-test for %s in %s" %(element, group))
            print("-------------------")
            control = dframe[dframe['test'] == 0]
            treatment = dframe[dframe['test'] == 1]
            control = control[control[group] == element]
            treatment = treatment[treatment[group] == element]
            print("%s control points available" %control.shape[0])
            print("%s treatment points available" %treatment.shape[0])
            ttest_unit(control['pages_visited'], treatment['pages_visited'])
    else:
        control = dframe[dframe['test'] == 0]
        treatment = dframe[dframe['test'] == 1]
        print("T-test for everything")
        ttest_unit(control['pages_visited'], treatment['pages_visited'])




if __name__ == "__main__":
    in_dir = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/EngagementTest/Data/Engagement_Test/'
    out_dir_plots = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/EngagementTest/Plots/'
    user_info_file = in_dir + 'user_table.csv'
    test_info_file = in_dir + 'test_table.csv'
    user_df = data_extraction(user_info_file)
    test_df = data_extraction(test_info_file)
    # diagnostics(user_df, ['signup_date'])
    # diagnostics(test_df, ['date', 'browser'])
    merged_df = merging(test_df, user_df, 'signup_date')
    df_new = new_feature(merged_df, 'signup_date', 'date')
    # simple_histograms(df_new, ['browser', 'tenure', 'date'])
    # simple_histograms(df_new, ['browser', 'tenure'], zoom=[0., 1000.])
    avg_pages(df_new, ['browser', 'tenure_group', 'date'], ['tenure'])
    t_tests(df_new, 'all')
    t_tests(df_new, 'browser')
    t_tests(df_new, 'tenure_group')