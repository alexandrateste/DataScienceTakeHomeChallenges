import pandas as pd
import numpy as np
from numpy import diff
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import DBSCAN

import datetime
import pytz
from pytz import timezone
from tzwhere import tzwhere
from pyzipcode import ZipCodeDatabase
from haversine import haversine

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from datetime import datetime


from tzwhere import tzwhere
tzwhere = tzwhere.tzwhere()
# Takes 10 sec !!!

# %matplotlib

plt.ion()


def data_extraction(file, column_names, categorical_var_list):
    if 'zipcode' in column_names:
        data_types = {"credit_card": np.int64,"date": str,
                      "transaction_dollar_amount": np.float64,
                      "Long": np.float64,"Lat": np.float64
                      }
    else:
        data_types = {"credit_card": np.int64,
                      "city": str, "state": str,
                      "zipcode": str,
                      "credit_card_limit": np.int64
                      }
    dframe = pd.read_csv(file, dtype=data_types)
    print "Dimensions: ", dframe.shape
    print("Summary: %s" %dframe.describe())
    for col in dframe.columns:
        if col not in categorical_var_list:
            print("Non empty rows for %s: %s" %(col, np.sum(np.isfinite(dframe[col].ravel()))))
        else:
            subset = dframe[dframe[col] != '']
            print "Non empty rows for ", col, ':', subset.shape[0]

    return dframe


def merger(card, transaction):
    merged = pd.merge(transaction, card, left_on='credit_card', right_on='credit_card', how='left')

    return merged


def new_features(df0, file):

    if not os.path.exists(file):
        print("Creating new features")
        dframe = df0.copy()
        dframe['date'] = pd.to_datetime(dframe['date'])
        unique_zipcodes = list(set(dframe['zipcode']))
        coordinates_dict = {}
        timezones_dict = {}
        print("Creating zipcode dictionary")
        for uzip in unique_zipcodes:
            coordinates_dict[uzip] = zipcode_coordinates(uzip)
        print("Creating timezone dictionary")
        for tpl in coordinates_dict.values():
            timezones_dict[tpl] = timezone_extraction(tpl)

        print("Extracting coordinates")
        dframe['Coordinates'] = [coordinates_dict[x] for x in dframe['zipcode']]

        print("Extracting timezones")
        dframe['Timezone'] = [timezones_dict[x] for x in dframe['Coordinates']]
        dframe.ix[dframe['zipcode'] == '94101', 'Timezone'] = 'America/Los_Angeles'
        dframe.ix[dframe['zipcode'] == '96801', 'Timezone'] = 'Pacific/Honolulu'
        # Info from: http://www.statoids.com/tus.html

        print("Computing local time")
        dframe['Local time'] = [(dframe.ix[x, 'date']).replace(tzinfo=timezone('UTC'))
                                    .astimezone(timezone(dframe.ix[x, 'Timezone']))
                                for x in range(0, dframe.shape[0])]
        # ix works here because the index is still the row number

        print("Creating lat/long tuples")
        dframe['Lat_Long'] = [(lat, lon) for lat, lon in zip(dframe['Lat'], dframe['Long'])]

        print("Computing distance")
        dframe['Distance (km)'] = [haversine(latlong, coord) for latlong, coord
                                   in zip(dframe['Lat_Long'], dframe['Coordinates'])]
        # Help from: https://pypi.python.org/pypi/haversine --> distance in km

        print("Adding day of year")
        dframe['Day of year'] = [x.timetuple().tm_yday for x in dframe['Local time']]

        dframe.to_csv(file, index=False)
    else:
        dframe = pd.read_csv(file)

    return dframe


def zipcode_coordinates(zipcd):
    print zipcd
    zcdb = ZipCodeDatabase()
    try:
        zipcode = zcdb[zipcd]
        coordinates = (zipcode.latitude, zipcode.longitude)
    except IndexError:
        if zipcd == '60290':
            coordinates = (41.881, -87.6247)
        else:
            coordinates = (np.nan, np.nan)

    return coordinates


def timezone_extraction(coordinates):
    print("Extracting timezones")
    timezone_name = tzwhere.tzNameAt(coordinates[0], coordinates[1])
    # Help from: http://stackoverflow.com/questions/15742045/getting-time-zone-from-lat-long-coordinates

    return timezone_name


def never_above(df0):
    print("Extracting list of well behaving clients")
    dframe = df0.copy()

    dframe['Local time'] = pd.to_datetime(dframe['Local time'])
    dframe['Month'] = dframe['Local time'].apply(lambda x: x.month)

    gp_card_month = dframe.groupby(['credit_card', 'Month', 'credit_card_limit'])['transaction_dollar_amount'].sum().reset_index()
    gp_card_month['Above threshold'] = [1 if total > thresh else 0 for total, thresh
                                        in zip(gp_card_month['transaction_dollar_amount'],
                                               gp_card_month['credit_card_limit'])]
    gp_month_only = gp_card_month.groupby('credit_card')['Above threshold'].sum()

    good_clients = list(gp_month_only[gp_month_only > 0].index)
    # List of credit card IDs for people who never went above their monthly limit

    print('Good clients are: %s' %good_clients)

    return good_clients


def offenders(df0):
    # Returns a dataframe with each date and the corresponding list of credit card IDs which limit was overcome that day
    print("Extracting list of daily offenders")
    dframe = df0.copy()
    dframe['Local time'] = pd.to_datetime(dframe['Local time'])
    dframe['Date only'] = dframe['Local time'].apply(lambda x: x.date())
    dframe['Month'] = dframe['Local time'].apply(lambda x: x.month)
    dframe = dframe.sort_values(['Local time'])  # Sorts records by date
    cumulative_spend = dframe.groupby(by=['credit_card', 'Month', 'Date only',
                                          'credit_card_limit'])['transaction_dollar_amount']\
                                      .sum().groupby(level=[0, 1]).cumsum().reset_index()
    # Computes the cumulative sum on both the credit card and the month,
    # i.e. equivalent to resetting the cumsum to 0 for each credit card and each month
    cumulative_spend = cumulative_spend.rename(columns={'transaction_dollar_amount': 'Cumulative spend'})
    cumulative_spend['Above threshold'] = [1 if total > thresh else 0
                                           for total, thresh in zip(cumulative_spend['Cumulative spend'],
                                                                    cumulative_spend['credit_card_limit'])]
    # print("cumulative_spend")
    # print cumulative_spend.head()

    offense_day = cumulative_spend.groupby(by=['credit_card', 'Month', 'Date only'])['Above threshold'].sum().groupby(level=[0, 1]).cumsum().reset_index()
    offense_day = offense_day.rename(columns={'Above threshold': 'Cumul threshold'})
    # print("Offense day")
    # print offense_day.head()

    first_day = offense_day[offense_day['Cumul threshold'] == 1]
    # print("first_day")
    # print first_day.head()

    offenders_per_day = first_day.groupby(['Date only'])['credit_card'].apply(lambda x: list(x)).reset_index()
    # print("offenders_per_day")
    # print offenders_per_day.head()

    return offenders_per_day


def plots(df0, out_dir):

    dframe = df0.copy()
    dframe['Local time'] = pd.to_datetime(dframe['Local time'])
    dframe['Local hour'] = dframe['Local time'].apply(lambda x: x.hour)
    gpdaycnt = dframe.groupby(['Day of year'])['transaction_dollar_amount'].count()
    gpdaysum = dframe.groupby(['Day of year'])['transaction_dollar_amount'].ssum()

    plt.figure(10)
    dframe.plot.scatter('Distance (km)', 'transaction_dollar_amount')
    plt.savefig(out_dir + 'AmountSpent_vs_Distance_AllDistances_2.png')
    plt.close()

    plt.figure(10)
    dframe['Distance bins (km)'] = ''
    dframe.ix[dframe['Distance (km)'] < 100., 'Distance bins (km)'] = '0-100km'
    dframe.ix[(dframe['Distance (km)'] >=100.) & (dframe['Distance (km)'] < 2500.), 'Distance bins (km)'] = '100-2500km'
    dframe.ix[dframe['Distance (km)'] >=2500., 'Distance bins (km)'] = '2500+km'

    sns.stripplot(x=dframe['Distance bins (km)'], y=dframe['transaction_dollar_amount'], jitter=.3, alpha=.05, size=2)
    g = sns.violinplot(x=dframe['Distance bins (km)'], y=dframe['transaction_dollar_amount'], inner=None, color="white", cut=0, scale="count")
    g.set(ylim=(-10., 1200.))
    # Help from: http://stackoverflow.com/questions/29812901/python-violin-plots

    plt.savefig(out_dir + 'AmountSpent_vs_Distance_AllDistances_ViolinPlots_ScatterPlot_2.png')
    plt.close()

    plt.figure(10)
    dframe['Distance (km)'].plot(kind='hist', bins=100, log=True)
    plt.xlabel('Distance (km)')
    plt.savefig(out_dir + 'Distance_Distribution_2.png')
    plt.close()

    plt.figure(10)
    dframe['transaction_dollar_amount'].plot(kind='hist', bins=100, log=True)
    plt.xlabel('Individual amounts spent')
    plt.savefig(out_dir + 'AmountSpent_Distribution_2.png')
    plt.close()

    plt.figure(10)
    dframe.plot.scatter('Local hour', 'transaction_dollar_amount')
    plt.savefig(out_dir + 'LocalHour_vs_Distance_AllDistances_ScatterPlot_2.png')
    plt.close()

    plt.figure(10)
    sns.stripplot(x=dframe['Local hour'], y=dframe['transaction_dollar_amount'], jitter=.3, alpha=.05, size=2)
    sns.violinplot(x=dframe['Local hour'], y=dframe['transaction_dollar_amount'], inner=None, color="white", cut=0, scale="count")
    plt.savefig(out_dir + 'LocalHour_vs_Distance_AllDistances_ViolinPlots_2.png')
    plt.close()

    plt.figure(10)
    gpdaycnt.plot()
    plt.ylabel('# of transactions')
    plt.savefig(out_dir + 'NumberOfTransactions_by_DayOfYear_2.png')
    plt.close()

    plt.figure(10)
    gpdaysum.plot()
    plt.ylabel('# of transactions')
    plt.savefig(out_dir + 'TotalAmountSpent_by_DayOfYear_2.png')
    plt.close()


def data_preparation(df0, list_to_drop):
    print("Preparing the data for clustering")
    dframe0 = df0.copy()
    dframe0['Local time'] = pd.to_datetime(dframe0['Local time'])

    dframe = dframe0.sample(frac=0.05, random_state=2345)
    local_time = dframe['Local time']

    # Should be part of the new_features function
    dframe['Local hour'] = dframe['Local time'].apply(lambda x: x.hour)
    dframe['zipcode_int'] = dframe['zipcode'].apply(lambda x: int(x))
    list_to_drop += ['zipcode']
    dframe = dframe.drop(list_to_drop, axis=1)
    # dframe_extended = pd.get_dummies(dframe)
    print("Features left: %s" %list(dframe.columns))

    matr = dframe.as_matrix()
    matr = StandardScaler().fit_transform(matr)  # Normalization!!!!!

    return dframe, matr, local_time


def clustering(original_df, matr, n_clusters):
    print("Identifying the clusters")
    y_pred = KMeans(n_clusters=n_clusters, random_state=1234).fit_predict(matr)
    check(original_df, y_pred, n_clusters, 'kmeans')

    return y_pred


def kmean_loop(original_df, matr, max_nbr_clusters, local_time):
    ori_df = original_df.copy()
    ori_df['Local time'] = local_time
    for k in range(2, max_nbr_clusters+1):
        print(">>> Processing for %s clusters <<<" %k)
        y_pred = clustering(ori_df, matr, k)

        sil(matr, y_pred, k)


def dbscan_initial(original_df, local_time):
    print("Running DBSCAN")
    # dframe = original_df.copy()
    sample_df = original_df.copy()
    # local_time = sample_df['Local time']
    # sample_df = sample_df.drop(list_to_drop, axis=1)
    matr = sample_df.as_matrix()

    db = DBSCAN(eps=1.4, min_samples=2).fit(matr)  # Pseudo grid search done in dbscan_loop
    #16, 1 -- 90, 0.6 // 5, 3 -- 35, 1.4
    # 18, 3

    labels = db.labels_

    # Number of clusters in labels, ignoring noise (-1) if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    sample_df['Local time'] = local_time
    check(sample_df, labels, n_clusters_, 'dbscan')

    return labels


def check(df0, clusters, nbr_clusters, method):
    dframe = df0.copy()
    dframe['Clusters'] = clusters
    dframe['Local time'] = pd.to_datetime(dframe['Local time'])
    dframe['Local hour'] = dframe['Local time'].apply(lambda x: x.hour)

    colors = {0: 'blue', 1: 'green', 2: 'red', 3: 'cyan', 4: 'purple', 5: 'gold',
              6:'orange', 7: 'pink', 8: 'brown', 9: 'magenta', 10: 'salmon', -1: 'None'}
    dframe['colors'] = [colors[x] for x in dframe['Clusters']]

    plt.figure(10)
    plt.scatter(dframe['Distance (km)'], dframe['transaction_dollar_amount'], c=dframe['colors'])
    # sns.lmplot('Distance (km)', 'transaction_dollar_amount', data=dframe, hue='Clusters', fit_reg=False)
    plt.xlabel('Distance (km)')
    plt.ylabel('Transaction amount')
    plt.savefig(out_dir + method + '_Distance_vs_AmountSpent_Clusters_'+str(nbr_clusters)+'clusters_2.png')
    plt.close()

    plt.figure(10)
    plt.scatter(dframe['Distance (km)'], dframe['transaction_dollar_amount'], c=dframe['colors'])
    # sns.lmplot('Distance (km)', 'transaction_dollar_amount', data=dframe, hue='Clusters', fit_reg=False)
    plt.xlabel('Distance (km)')
    plt.ylabel('Transaction amount')
    plt.xlim([-5., 20.])
    plt.savefig(out_dir + method + '_Distance_vs_AmountSpent_Clusters_'+str(nbr_clusters)+'clusters_ZoomIn_2.png')
    plt.close()

    plt.figure(10)
    # sns.lmplot('Local hour', 'transaction_dollar_amount', data=dframe, hue='Clusters', fit_reg=False)
    plt.scatter(dframe['Local hour'], dframe['transaction_dollar_amount'], c=dframe['colors'])
    plt.xlabel('Local hour')
    plt.ylabel('Transaction amount')
    plt.savefig(out_dir + method + 'LocalHour_vs_AmountSpent_Clusters_'+str(nbr_clusters)+'clusters_2.png')
    plt.close()

    plt.figure(10)
    plt.scatter(dframe['Distance (km)'], dframe['Local hour'], c=dframe['colors'])
    # sns.lmplot('Distance (km)', 'Local hour', data=dframe, hue='Clusters', fit_reg=False)
    plt.xlabel('Distance (km)')
    plt.ylabel('Local hour')
    plt.savefig(out_dir + method + 'Distance_vs_LocalHour_Clusters_'+str(nbr_clusters)+'clusters_2.png')
    plt.close()

    plt.figure(10)
    plt.scatter(dframe['Distance (km)'], dframe['Local hour'], c=dframe['colors'])
    # sns.lmplot('Distance (km)', 'Local hour', data=dframe, hue='Clusters', fit_reg=False)
    plt.xlabel('Distance (km)')
    plt.ylabel('Local hour')
    plt.xlim([-5., 20.])
    plt.savefig(out_dir + method + 'Distance_vs_LocalHour_Clusters_'+str(nbr_clusters)+'clusters_ZoomIn_2.png')
    plt.close()


def dbscan_loop(matr4clustering):
    small_matr = matr4clustering[np.random.choice(matr4clustering.shape[0], 25000, replace=False), :]
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=len(small_matr)).fit(small_matr)
    distances, indices = nbrs.kneighbors(small_matr)
    for kk in range(0, len(small_matr)):
        plt.plot(sorted(distances[:, kk]))

    eps_range = np.arange(0.2, 10, 0.4)
    minpts_range = range(10, 100, 5)

    nbr_clusters = np.zeros(shape=(len(eps_range), len(minpts_range)))
    for ep in eps_range:
        for mp in minpts_range:
            db = DBSCAN(eps=ep, min_samples=mp).fit(small_matr)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            nbr_clusters[list(eps_range).index(ep), list(minpts_range).index(mp)] = n_clusters_
            print("Eps=%s, minPts=%s, %s clusters" % (ep, mp, n_clusters_))

    graph = plt.matshow(nbr_clusters, cmap=plt.cm.jet)
    plt.colorbar(graph)
    plt.clim(0, 10)

    # %run credit_card_transactions.py > alloutput.txt  # to save stdout to file
    # minpts, eps = 16, 1 or 5, 3


def saving_data(df0, in_dir):
    bla = df0.copy()
    bla['clusters'] = clusters
    bla['Local time'] = pd.to_datetime(bla['Local time'])
    bla['Local hour'] = bla['Local time'].apply(lambda x: x.hour)
    colors = {0: 'blue', 1: 'green', 2: 'red', 3: 'cyan', 4: 'purple'}
    bla['colors'] = [colors[x] for x in bla['clusters']]
    bla.to_csv(in_dir+'Data_With_Clusters.csv')


def sil(matr, y_pred, n_clusters):
    print("Computing the silhouette scores")
    silhouette_avg = silhouette_score(matr, y_pred)
    print("Overall silhouette average: %s" %silhouette_avg)
    sample_silhouette_values = silhouette_samples(matr, y_pred)

    plt.figure(20)
    y_lower = 10
    for i in range(n_clusters):
        print("Cluster # %s" %i)
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[y_pred == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    plt.savefig(out_dir + 'Silhouette_Plot_'+str(n_clusters)+'clusters_2.png')
    plt.close()


# def threedplot(df0):
#     # !!! HAS TO BE RUN IN A TOTALLY DIFFERENT WINDOW FROM ALL THE REST, OTHERWISE DOESN'T PLOT ANYTHING !!!
#     from matplotlib import pyplot
#     import pylab
#     from mpl_toolkits.mplot3d import Axes3D
#     fig = pylab.figure()
#     ax = Axes3D(fig)
#     ax.scatter(list(df0['Distance (km)']), list(df0['transaction_dollar_amount']), list(df0['Local hour']))
#     pyplot.show()
    # ax.scatter(list(df0['Distance (km)']), list(df0['transaction_dollar_amount']), list(df0['Local hour']),
    #            c=list(df0['colors']))
#     help from: http://stackoverflow.com/questions/1985856/how-to-make-a-3d-scatter-plot-in-python


if __name__ == "__main__":
    in_dir = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/CreditCardTransactions/Data/'
    out_dir = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/CreditCardTransactions/Plots/'
    cc_file = in_dir + 'cc_info.csv'
    trans_file = in_dir + 'transactions.csv'
    new_feat_file = in_dir + 'credit_card_transactions_with_new_features.csv'
    card_info_df = data_extraction(cc_file, [], ['city', 'state', 'zipcode'])
    transactions_info_df = data_extraction(trans_file, [], ['date'])
    merged_df = merger(card_info_df, transactions_info_df)
    with_new_feat = new_features(merged_df, new_feat_file)
    # good_clients = never_above(with_new_feat)
    # offenders_per_day = offenders(with_new_feat)
    # plots(with_new_feat, out_dir)
    list_to_drop = ['date', 'Long', 'Lat', 'city', 'state', 'Coordinates', 'Timezone', 'Local time', 'Lat_Long']
    with_new_feat, matr4clustering, local_time = data_preparation(with_new_feat, list_to_drop)
    # kmean_loop(with_new_feat, matr4clustering, 7, local_time)  #7
    # check(with_new_feat, clusters)
    # saving_data(with_new_feat, in_dir)
    labels = dbscan_initial(with_new_feat, local_time)
    # dbscan_loop(matr4clustering)


""" >>>> IMPLEMENT DBSCAN <<<< """
# USE: % matplotlib before plotting on the prompt

