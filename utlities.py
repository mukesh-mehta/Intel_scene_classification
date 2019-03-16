#Prepare data for 10fold classification
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

def stratified_split(train_file,path_to_save_metadata, num_splits=10, random_state=2):
    train_data = pd.read_csv(train_file)
    skf = StratifiedKFold(n_splits=num_splits,random_state=random_state)
    i=0
    for train_idx, test_idx in skf.split(train_data["image_name"],train_data["label"]):
        train_data.loc[train_idx].to_csv("{}/train_{}.csv".format(path_to_save_metadata,i), index=False)
        train_data.loc[test_idx].to_csv("{}/val_{}.csv".format(path_to_save_metadata,i), index=False)
        i+=1
    return