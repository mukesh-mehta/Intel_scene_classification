#Prepare data for 10fold classification
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split

def stratified_split(train_file,path_to_save_metadata, keep_train_dev=True, num_splits=10, random_state=2):
    train_data = pd.read_csv(train_file)
    if keep_train_dev:
    	train_data, train_dev_data = train_test_split(train_data, test_size=0.1, stratify = train_data["label"])
    	train_dev_data.to_csv("{}/train_dev.csv".format(path_to_save_metadata),index=False)
    train_data.reset_index(inplace=True,drop=True)
    skf = StratifiedKFold(n_splits=num_splits,random_state=random_state)
    i=0
    for train_idx, test_idx in skf.split(train_data["image_name"],train_data["label"]):
        train_data.loc[train_idx].to_csv("{}/train_{}.csv".format(path_to_save_metadata,i), index=False)
        train_data.loc[test_idx].to_csv("{}/val_{}.csv".format(path_to_save_metadata,i), index=False)
        i+=1
    return