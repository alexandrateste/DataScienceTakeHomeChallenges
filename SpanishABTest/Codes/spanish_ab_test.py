import pandas as pd
import numpy as np
import csv

import matplotlib.pyplot as plt
from scipy import stats

plt.ion()

def data_extraction(testfile, userfile):
    """

    :param testfile:
    :param userfile:
    :return:
    """
    test_df = pd.read_csv(testfile)
    user_df = pd.read_csv(userfile)
    print test_df.shape
    print user_df.shape
    merged_df = pd.merge(left=test_df, right=user_df, left_on='user_id', right_on='user_id', how='inner')

    return merged_df


def groupings(df0):
    """

    :param df0:
    :return:
    """
    dframe = df0.copy()
    language = dframe.groupby(['browser_language']).count()
    country = dframe.groupby(['country']).count()
    print language
    print country


def data_of_interest(df0, confidence_level):
    """

    :param df0:
    :return:
    """
    dframe = df0.copy()
    final_df = dframe[(dframe.country != 'Spain') & (dframe.browser_language == 'ES')]
    # final_df = dframe[dframe.browser_language == 'ES']
    # Disregard any languages different from Spanish and Spain inhabitants because they would never be in the treatment group
    control = final_df[final_df.test == 0]
    treatment = final_df[final_df.test == 1]
    conversions_spaniard = control.conversion.sum()
    conversions_others = treatment.conversion.sum()
    # print("Conversions for control %s and treatment %s" %(conversions_spaniard, conversions_others))

    # By hand:
    # n_control = control.shape[0]
    # n_treatment = treatment.shape[0]
    # proba_control = 1.*conversions_spaniard/n_control  # let's assume
    # proba_treatment = 1.*conversions_others/n_treatment  # let's assume
    # std_control = np.sqrt(proba_control*(1.-proba_control))
    # std_treatment = np.sqrt(proba_treatment*(1.-proba_treatment))
    # denominator = np.sqrt((std_control**2)/n_control + (std_treatment**2)/n_treatment)
    # numerator = control['conversion'].mean() - treatment['conversion'].mean()
    # ratio = numerator / denominator
    # print("ratio = %s" %ratio)
    #
    # # With built-in function:
    # tstat, pvalue = stats.ttest_ind(control['conversion'], treatment['conversion'], equal_var=False)
    # pvalue = 2.*pvalue  # because function returns 2-sided, while we want 1-sided
    # print("Overall T-statistics: %s and P-value: %s" %(tstat, pvalue))

    # By country:
    for ctry in list(set(final_df.country)):
        cont = control[control.country == ctry]
        treat = treatment[treatment.country == ctry]
        tstat, pvalue = stats.ttest_ind(cont['conversion'], treat['conversion'], equal_var=False)
        pvalue = 2.*pvalue  # because function returns 2-sided, while we want 1-sided
        print("For %s T-statistics: %s and P-value: %s" %(ctry, tstat, pvalue))

    zscore = stats.norm.ppf(confidence_level)

    # for overall or final tstat
    if tstat > zscore:
        print("Number of conversions for control group > for treatment group")
        return True
    else:
        print("Number of conversions for control group < for treatment group")
        return False


if __name__ == "__main__":
    in_dir = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/' \
               'SpanishABTest/Data/Translation_Test/'
    testfile = in_dir + 'test_table.csv'
    userfile = in_dir + 'user_table.csv'

    merged_df = data_extraction(testfile, userfile)
    # print merged_df.columns
    # print merged_df.head()
    # print merged_df.describe()
    verdict = data_of_interest(merged_df, 0.95)
    print verdict
