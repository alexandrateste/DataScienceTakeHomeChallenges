import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import statsmodels.tsa.stattools
import statsmodels.graphics.tsaplots
from statsmodels import api as sm
from scipy import stats

plt.ion()


def data_extraction(file, collist):
    """

    :param file:
    :return:
    """
    dframe = pd.read_csv(file)
    print("Dimensions:")
    print(dframe.shape)

    print("Columns: %s" %dframe.columns)
    print("Summary: %s" %dframe.describe())

    if dframe.shape[0]:
        for col in dframe.columns:
            if col not in collist:
                length = np.sum(np.isfinite(dframe[col]))
            else:
                length = len([x for x in dframe[col] if x != ''])
            print("%s %% of non-nan records for %s" %(100.*length/dframe.shape[0], col))
    else:
        print("This dataframe is empty")

    dframe['date'] = [datetime.strptime(x, "%Y-%m-%d").date() for x in dframe['date']]

    return dframe


def diagnostics(df0, collist):
    """

    :param df0:
    :param collist:
    :return:
    """
    dframe = df0.copy()
    for col in dframe.columns:
        print("Plotting for %s" %col)
        filename = out_dir_plots + 'Distribution_' + col + '.png'
        if not os.path.exists(filename):
            if col not in collist:
                if col == 'ad':
                    grouped = dframe.groupby('ad')['date'].count()
                    grouped.plot(kind='bar')
                    plt.ylabel("Number of days")
                    plt.savefig(filename)
                    plt.close()
                else:
                    dframe[col].plot(kind='hist', bins=100)
                    plt.xlabel(col)
                    plt.savefig(filename)
                    plt.close()


def time_series_plots(df0, list_of_interest):
    dframe = df0.copy()
    dframe['date'] = pd.to_datetime(dframe['date'])
    dframe = dframe.set_index('date')
    for element in list_of_interest:
        print("Time series for %s" %element)
        filename = out_dir_plots + 'Time_series_' + element + '.png'
        if not os.path.exists(filename):
            master_df = pd.DataFrame([])
            for ad in list(set(dframe['ad'])):
                subset = dframe[dframe['ad'] == ad]
                subset = pd.DataFrame(subset[element])
                subset.columns = [ad]
                master_df = pd.concat([master_df, subset], axis=1)

            master_df.plot(figsize=(24,8))
            plt.ylabel(element)
            plt.title('Evolution over time')
            plt.legend(prop={'size':8})
            plt.savefig(filename)
            plt.close()


def new_features(df0):
    dframe = df0.copy()
    dframe['percent_click'] = 100.*dframe['clicked'] / dframe['shown']
    dframe['percent_conversion'] = 100.*dframe['converted'] / dframe['clicked']

    return dframe


def ranking(df0, group, method):
    dframe = df0.copy()

    for gp in group:
        print("Plotting %s graph for %s" %(method, gp))
        filename = out_dir_plots + 'Ad_group_ranking_by_' + gp + '_' + method + '.png'
        if not os.path.exists(filename):
            if method == 'cumulative':
                grouped = pd.DataFrame(dframe.groupby('ad')[gp].sum())
            elif method == 'average':
                grouped = pd.DataFrame(dframe.groupby('ad')[gp].mean())

            grouped = grouped.sort_values([gp], ascending=False)
            grouped.index = [x.replace('ad_group_', '') for x in grouped.index]
            grouped.plot(kind='bar')
            plt.ylabel(method.capitalize() + ' ' + gp)
            plt.savefig(filename)
            plt.close()


def plot_time_series(df0, element):
    dframe = df0.copy()

    filename = out_dir_plots + 'All_ad_groups_time_series_' + element + '.png'
    if not os.path.exists(filename):
        pivoted = dframe.pivot('date', 'ad', element)
        pivoted.index = pd.to_datetime(pivoted.index)
        pivoted.plot(figsize=(24,8))
        plt.legend(loc='upper left', prop={'size': 8})
        plt.savefig(filename)
        plt.close()


def smoothed_tseries(df0, element):
    dframe = df0.copy()

    pivoted = dframe.pivot('date', 'ad', element)
    pivoted.index = pd.to_datetime(pivoted.index)
    master_smoothed = pd.DataFrame([])
    slope_df = pd.DataFrame([])
    for ad in pivoted.columns:
    # for ad in ['ad_group_6', 'ad_group_7']:
        filename = out_dir_plots + 'All_ad_groups_time_series_smoothed' + element + '_' + ad + '.png'
        if not os.path.exists(filename):
            ax = pivoted[ad].plot(figsize=(24,8))
            smoothed = pd.DataFrame(pd.rolling_median(pivoted[ad], window=10, center=True))
            smoothed.plot(ax=ax, linestyle='-.')
            plt.legend(loc='upper left', prop={'size': 8})
            plt.savefig(filename)
            plt.close()
            smoothed.columns = [ad]
            master_smoothed = pd.concat([master_smoothed, smoothed], axis=1)
            smoothed.index = pd.to_datetime(smoothed.index)
            smoothed['delta_days'] = [(x-smoothed.index[0]).total_seconds()/86400. for x in smoothed.index]
            print smoothed.head()
            smoothed = smoothed[np.isfinite(smoothed[ad])]
            result = np.polyfit(smoothed['delta_days'], smoothed[ad], 1)
            print("Parameters: %s" %result)
            if result[0] < 0.:
                grp = 'down'
            elif result[0] > 0.:
                grp = 'up'
            else:
                grp = 'flat'

            values = [result[0], grp]
            current_slope = pd.DataFrame(values).transpose()
            current_slope.column = ['slope', 'group']
            slope_df = slope_df.append(current_slope)

    slope_df.index = [pivoted.columns]
    slope_df.to_csv(in_dir + 'Ad_slopes_grouping.csv')


def plot_acf_pacf(df0):  # Doesn't converge on different ads
    dframe = df0.copy()
    pivoted = dframe.pivot('date', 'ad', 'shown')
    pivoted.index = pd.to_datetime(pivoted.index)
    for ad in pivoted.columns:
    # for ad in ['ad_group_1']:
        print("Processing ad #%s" %ad.split('_')[2])
        subset = pivoted[np.isfinite(pivoted[ad])]
        time_series_initial = pd.TimeSeries(subset[ad].ravel(), index=pd.to_datetime(subset.index))
        time_series = np.log(time_series_initial)
        filename = out_dir_plots + 'AutoCorrPlots_' + ad + '.png'
        if not os.path.exists(filename):
            # print time_series
            pa = sm.tsa.pacf(time_series)
            acf = sm.tsa.acf(time_series)
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(acf)
            z = stats.norm.ppf(0.99)
            n = time_series.shape[0]
            ax1.axhline(y=z / np.sqrt(n), linestyle='--', color='red')
            ax1.axhline(y=-z / np.sqrt(n), linestyle='--', color='red')
            ax1.set_ylabel('Auto-Corr Func.')
            ax1.set_title(ad)
            ax2.plot(pa)
            ax2.axhline(y=z / np.sqrt(n), linestyle='--', color='red')
            ax2.axhline(y=-z / np.sqrt(n), linestyle='--', color='red')
            ax2.set_ylabel('Partial Auto-Corr Func.')
            plt.savefig(filename)
            plt.close()

        filename = out_dir_plots + 'Prediction_' + ad + '.png'
        if not os.path.exists(filename):
            try:
                # Most of the plots show a 1 peak at lag =1 for ACF and for PACF --> model with params p=1, q=0
                # res10 = sm.tsa.ARMA(time_series, (1, 0)).fit()
                # res71 = sm.tsa.ARMA(time_series, (7, 1)).fit()
                # res81 = sm.tsa.ARMA(time_series, (8, 1)).fit()
                res51 = sm.tsa.ARMA(time_series, (5, 1)).fit()
                # res121 = sm.tsa.ARMA(time_series, (12, 1)).fit()
                fig, ax = plt.subplots()
                ax = time_series.ix['2015-10-01':].plot(ax=ax)
                # fig = res10.plot_predict('2015-11-23', '2015-12-16', dynamic=True, ax=ax, plot_insample = False)
                # fig = res20.plot_predict('2015-11-23', '2015-12-16', dynamic=True, ax=ax, plot_insample = False)
                # fig = res11.plot_predict('2015-11-23', '2015-12-16', dynamic=True, ax=ax, plot_insample = False)
                # fig = res81.plot_predict('2015-11-23', '2015-12-16', dynamic=True, ax=ax, plot_insample = False)
                fig = res51.plot_predict('2015-11-23', '2015-12-16', dynamic=True, ax=ax, plot_insample = False)
                # fig = res121.plot_predict('2015-11-23', '2015-12-16', dynamic=True, ax=ax, plot_insample = False)
                # fig2, ax2 = plt.subplots()
                # y_resid81 = res81.resid
                # y_resid91 = res91.resid
                # y_resid121 = res121.resid
                # y_resid81.plot()
                # y_resid91.plot()
                # y_resid121.plot()
            except ValueError:
                continue

            plt.savefig(filename)
            plt.close()







            # Relevant information from https://bicorner.com/2015/11/16/time-series-analysis-using-ipython/:
#
# Once we determine the nature of the auto-correlations we use the following rules of thumb.
#
# - Rule 1: If the ACF shows exponential decay, the PACF has a spike at lag 1, and no correlation for other lags, then use one autoregressive (p)parameter
# - Rule 2: If the ACF shows a sine-wave shape pattern or a set of exponential decays, the PACF has spikes at lags 1 and 2, and no correlation for other lags, the use two autoregressive (p) parameters
# - Rule 3: If the ACF has a spike at lag 1, no correlation for other lags, and the PACF damps out exponentially, then use one moving average (q) parameter.
# - Rule 4: If the ACF has spikes at lags 1 and 2, no correlation for other lags, and the PACF has a sine-wave shape pattern or a set of exponential decays, then use two moving average (q) parameter.
# - Rule 5: If the ACF shows exponential decay starting at lag 1, and the PACF shows exponential decay starting at lag 1, then use one autoregressive (p) and one moving average (q) parameter.

        # def main():
    #     prototype = np.random.random(60)
    #     for _ in xrange(560 / 60):
    #         prototype = np.concatenate((prototype, np.random.normal(0, 0.1, 60) + prototype[:60]))
    #     prototype = prototype[60:]
    #     n = prototype.shape[0]
    #     pa = sm.tsa.pacf(prototype, 100)
    #     acf = sm.tsa.acf(prototype, nlags=100)
    #
    #
    #
    #
    # df = pd.read_csv(csv_file, index_col=[0], sep='\t')
    # grouped = df.groupby('adserver_id')
    # group = list(grouped)[0][1]
    #
    # ts_data = pd.TimeSeries(group.c_start.values, index=pd.to_datetime(group.day))
    # # positive-valued process, looks non-stationary
    # # simple way is to do a log transform
    # fig, axes = plt.subplots(figsize=(10,8), nrows=3)
    # ts_data.plot(ax=axes[0])
    #
    # ts_log_data = np.log(ts_data)
    # ts_log_data.plot(ax=axes[1], style='b-', label='actual')
    #
    # # in-sample fit
    # # ===================================
    # model = sm.tsa.ARMA(ts_log_data, order=(1,1)).fit()
    # print(model.params)
    #
    # y_pred = model.predict(ts_log_data.index[0].isoformat(), ts_log_data.index[-1].isoformat())
    # y_pred.plot(ax=axes[1], style='r--', label='in-sample fit')
    #
    # y_resid = model.resid
    # y_resid.plot(ax=axes[2])
    #
    # # out-sample predict
    # # ===================================
    # start_date = ts_log_data.index[-1] + Day(1)
    # end_date = ts_log_data.index[-1] + Day(7)
    #
    # y_forecast = model.predict(start_date.isoformat(), end_date.isoformat())
    #
    # print(y_forecast)
    #
    # # NOTE: this step introduces bias, it is used here just for simplicity
    # # E[exp(x)] != exp[E[x]]
    # print(np.exp(y_forecast))
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #


if __name__ == '__main__':
    in_dir = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/AdAnalysis/Data/'
    out_dir_plots = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/AdAnalysis/Plots/'
    file = in_dir + 'ad_table.csv'
    df = data_extraction(file, ['date', 'ad'])
    print("bla")
    diagnostics(df, ['date'])
    df_new = new_features(df)
    time_series_plots(df_new, ['shown', 'clicked', 'converted', 'avg_cost_per_click',
                               'total_revenue', 'percent_click', 'percent_conversion'])
    ranking(df_new, ['total_revenue', 'clicked', 'converted'], 'cumulative')
    ranking(df_new, ['total_revenue', 'clicked', 'converted'], 'average')
    plot_time_series(df_new, 'shown')
    plot_time_series(df_new, 'avg_cost_per_click')
    # plot_acf_pacf(df_new)
    smoothed_tseries(df_new, 'avg_cost_per_click')