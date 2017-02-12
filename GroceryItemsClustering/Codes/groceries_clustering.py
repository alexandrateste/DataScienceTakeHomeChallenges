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

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def data_extraction(file):

    print("===========================")
    print("Extracting file %s" %file)
    print("===========================")
    dframe = pd.read_csv(file)
    print "Dimensions: ", dframe.shape
    print("Summary: %s" %dframe.describe())
    print dframe.head()

    return dframe


def biggest_buyer(df0):
    """

    :param df0: purchase dataframe
    :return: the user id of the buyer who bought the most items over her lifetime
    """
    dframe = df0.copy()
    unique_id = list(set(purchase_df['id']))
    print("There are %s unique users" %len(unique_id))

    dframe['item_counts'] = [len(x.split(',')) for x in purchase_df['id']]

    buyer = dframe.groupby(['user_id'])['item_counts'].sum()

    big_buy = buyer.argmax()
    print("%s bought the biggest number of items (%s) overall in her customer lifetime" %(big_buy,buyer.max()))

    return big_buy



def buyer_by_item(df0, items_df):
    dframe = df0.copy()
    # Check that all items in the purchase df are in the items df
    reference_items_list = list(set(items_df['Item_id']))
    purchase_list = list(set(dframe.sum()['id']))
    if len(reference_items_list) == len(purchase_list):
        print("All purchased items are captured in the item table")
    elif len(reference_items_list) > len(purchase_list):
        print("Some items were never purchased")
    else:
        print("Some items that were purchased are not in the items table")

    # dframe = dframe.set_index('user_id')
    filename = in_dir + 'buyers_with_items.csv'
    if not os.path.exists(filename):
        # One row per user -- each column for each item purchased
        dframe = dframe.groupby('user_id')['id'].apply(lambda x: "%s" % ','.join(x))
        # Combines all rows with same index together
        master_df = pd.DataFrame([])
        for row in dframe.index:
            print("Row: %s" %row)
            print dframe.ix[row, 'id']
            print type(dframe.ix[row, 'id'])
            content = (dframe.ix[row, 'id']).split(',')
            current_df = pd.DataFrame(np.ones(len(content))[np.newaxis], columns=[content])
            current_df.index = [row]

            master_df = pd.concat([master_df, current_df], axis=1)
        master_df = master_df.groupby(master_df.columns, axis=1).sum()
        # Combines repeated columns
        master_df = master_df.groupby(master_df.index).sum()
        # Combines the rows
        master_df.index.name = 'user_id'
        master_df.to_csv(filename)
    else:
        master_df = pd.read_csv(filename, index_col='user_id')


    buyers_filename = in_dir + 'items_biggest_buyers.csv'
    if not os.path.exists(buyers_filename):
        # One unique row / one column item ID
        user_by_item = pd.DataFrame(columns=master_df.columns)
        for col in master_df.columns:
            user_by_item[col] = master_df[col].argmax().ravel()
        user_by_item_transp = user_by_item.transpose()
        user_by_item_transp.index.name = 'items'
        print user_by_item_transp.head()
        user_by_item_transp.columns = ['Biggest_buyer']
        print user_by_item_transp.head()
        items_df = items_df.set_index('Item_id')
        items_df.index = [int(x) for x in items_df.index]
        user_by_item_transp.index = [int(x) for x in user_by_item_transp.index]
        print items_df
        combination = pd.merge(user_by_item_transp, items_df, left_index=True, right_index=True, how='left')
        print combination.head()
        combination = combination[['Item_name', 'Biggest_buyer']]
        print combination.head()

        combination.to_csv(buyers_filename)
    else:
        combination = pd.read_csv(buyers_filename)

    return combination


def co_occurrence_matrix(df0):
    dframe = df0.copy()

    master_dict = {}
    for itemslist in dframe['id']:
        # print("Row: %s" % row)
        content = itemslist.split(',')

        for element in list(set(content)):
            if element in master_dict.keys():
                master_dict[element] += content.count(element)
            else:
                master_dict[element] = [content.count(element)]




        master_df = pd.concat([master_df, current_df], axis=1)
    master_df = master_df.groupby(master_df.columns, axis=1).sum()
    # Combines repeated columns
    master_df = master_df.groupby(master_df.index).sum()

    # master_dict = {}
    # for idlist in dframe['id']:
    #     content = idlist.split(',')
    #     for element in list(set(content)):
    #         if element in master_dict.keys():
    #             master_dict[element] += content.count(element)
    #         else:
    #             master_dict[element] = [content.count(element)]
    #
    # # Assumption: one purchase contains only one occurrence of a given item
    #
    #     if len(master_dict) == 0:
    #
    #     current_df = pd.DataFrame(np.ones(len(content)), columns=[content])
    #
    #     master_df = pd.concat([master_df, current_df], axis=1)
    # master_df = master_df.groupby(master_df.columns, axis=1).sum()
    # # combines repeated columns
    #



    row=user, col = 1 per item (sum)




if __name__ == "__main__":
    in_dir = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/GroceryItemsClustering/Data/'
    out_dir_plots = '/Users/alexandra/ProgCode/DataScienceTakeHomeChallenges/GroceryItemsClustering/Plots/'
    items_file = in_dir + 'item_to_id.csv'
    history = in_dir + 'purchase_history.csv'

    purchase_df = data_extraction(history)
    identifiers_df = data_extraction(items_file)
    bbuyer = biggest_buyer(purchase_df)


