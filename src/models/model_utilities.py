import pandas as pd
import numpy as np
import re

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score,accuracy_score,precision_recall_curve,confusion_matrix,make_scorer
from sklearn.metrics import roc_auc_score, roc_curve

#set raw and interim data paths
raw_data_dir = '~/nba_dfs_value/data/raw/'
interim_data_dir= '~/nba_dfs_value/data/interim/'
processed_data_dir = '~/nba_dfs_value/data/processed/'

#read in full data file
ndf = pd.read_pickle(processed_data_dir+"full.pkl")

def choose_features(df,target,unused_features,non_lag_features):
    """
    Takes in fully transformed data frame and chooses relevant features using 
    feature correlations and recursive feature extraction.


    Parameters:
    df               (pandas df)       :     Fully transformed/engineered dataframe
    target           (list)            :     target variable
    unused_features   (list)            :     features not relevant to analysis
    non_lag_features  (list)            :     features that dont need to be lagged (known)

    Returns:
    [list]         :     model feature set
    """

    #save non_lag features
    non_lag_features_df = df[non_lag_features]

    #drop unused and non-lag features
    df.drop(columns = unused_features,axis=1,inplace=True)
    df.drop(columns = non_lag_features,axis=1,inplace=True)

    #find only lagged columns and make new data frame
    lag_columns = [x for x in df.columns if re.search("_lag",x)]
    lag_df = df[lag_columns].copy()

    #create feature correlation matrix and extract and then remove features with high correlations
    correlated_features = set()
    corr_matrix = lag_df.corr()
    for i in range(len(corr_matrix.columns)):
      for j in range(i):
          if abs(corr_matrix.iloc[i, j]) > 0.85:
              colname = corr_matrix.columns[i]
              correlated_features.add(colname)

    final_df = lag_df.drop(columns=correlated_features,axis=1)

    #add back relevant non-lagged features
    final_df = pd.concat([lag_df,non_lag_features_df],axis=1)
    features = final_df.columns

    #prepare feature and target set
    dummies = ['player_type_lag',
                'venue',
                'start',
                'new_start',
                'def_type_lag',
                'value_lag',
                'rest']

    target = 'value'
    X,y,model_columns = prepare_model_data(df,features,dummies,target)

    #create training and test splits
    X_train,X_ho,y_train,y_ho = tv_split(X,y,test=0.2)

    #make f.5 scorer to emphasize precision
    fhalf_scorer = make_scorer(fbeta_score, beta=.5)

    #create timeseries split generators
    tscv = TimeSeriesSplit(n_splits=5)

    #use recursive feature extraction to choose best features
    rfc = RandomForestClassifier(n_estimators = 100,max_dept = 6, min_samples_leaf=5,n_jobs=-1)
    rfecv = RFECV(estimator=rfc, step=12, cv=tscv, scoring=fhalf_scorer,verbose=True)
    rfecv.fit(X_train, y_train)
    X_train.drop(X_train.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

    return X_train.columns
    


def prepare_model_data(df,features,dummies,target):
    """
    Takes in fully transformed data frame and creates features set (with dummies)
    target, and feature column names for future use

    Parameters:
    df          (pandas df)       :     Fully transformed/engineered dataframe
    features    (list)            :     features to try
    dummies     (list)            :     list of features to create dummy variable for
    target      (list)            :     target variable

    Returns:
    (pandas df)       :     model feature set
    (pandas series)   :     target 
    (list)            :     feature columns

    """
    df_dummies = pd.get_dummies(df_model,columns=dummies)
    X = df_dummies[features]
    model_columns = X.columns
    y = df[target]

    return X,y,model_columns


def tv_split(X,y,test=.2):
    """
    Prepares dataframe for use in time series split

    Parameters:
    X           (pandas df)        :    feature dataframe
    y           (pandas series)    :    target value series
    test        (float)            :    % of data to use as holdout set

    Returns:
    (pandas df)         : training feature set
    (pandas df)         : holdout feature set
    (pandas series)     : training target set
    (pandas series)     : holdout target set
    """

    split = int(len(X)*(1-test))
    X_train = pd.DataFrame(data= X[:split].values, columns = X.columns)
    X_ho = pd.DataFrame(data= X[split:].values, columns = X.columns)
    y_train = y[:split].values
    y_ho =  y[split:].values
    
    return X_train,X_ho,y_train,y_ho




if __name__ == "__main__":

  raw_data_dir = '~/nba_dfs_value/data/raw/'
  interim_data_dir= '~/nba_dfs_value/data/interim/'
  processed_data_dir = '~/nba_dfs_value/data/processed/'

  df = pd.read_pickle(processed_data_dir+"full.pkl")


  target = 'value'
  unused_features = ['index','dataset','game_id','date','player_id','name','team','opp','fpd','last_start','pos',
                    'last_start_lag', 'new_start_lag']
  
  non_lag_features = ['rest','venue','start','value','salary','new_start','player_type_lag','def_type_lag']
  features = choose_features(df,target,unused_features,non_lag_features)

  dummies = ['player_type_lag','venue','start','new_start','def_type_lag','value_lag','rest']
  X,y,model_columns = prepare_model_data(df,features,dummies,target)

  X_train,X_ho,y_train,y_ho = tv_split(X,y,test=.02)

