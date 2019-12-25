import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit

#set raw and interim data paths
raw_data_dir = '~/nba_dfs_value/data/raw/'
interim_data_dir= '~/nba_dfs_value/data/interim/'
processed_data_dir = '~/nba_dfs_value/data/processed/'

#read in full data file
ndf = pd.read_pickle(processed_data_dir+"full.pkl")

features = ['value_lag',
                'sal_prev',
                'fp_3MA_lag',
                 'fp_10MA_lag',
                  'min_3MA_lag',
                  'min_10MA_lag',
                  'player_type_lag',
                  'venue',
                 'start',
                 'new_start',
                'def_type_lag',
                  'usage_3MA_lag',
                  'usage_10MA_lag',
                'rest'] 

dummies = ['value_lag',
           'player_type_lag',
            'venue',
            'start',
             'new_start',
            'def_type_lag',
            'rest']

target = ['value'] 

def prepare_model_data(df,features,dummies,target):
    df_model = df[features]
    X = pd.get_dummies(df_model,columns=dummies)
    model_columns = X.columns
    y = df[target]

    return X,y,model_columns

