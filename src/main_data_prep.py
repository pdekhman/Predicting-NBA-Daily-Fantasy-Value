import pandas as pd 
import numpy as np
from data.make_dataset import load_clean
from features.build_features import build_features
from features.kmeans_utilities import kmean_players_df, kmean_defenses_df


#set raw and interim data paths
raw_data_dir = '~/nba_dfs_value/data/raw/'
interim_data_dir= '~/nba_dfs_value/data/interim/'
processed_data_dir = '~/nba_dfs_value/data/processed/'

#read player file
player_file = raw_data_dir+'NBA1819.csv'
player= pd.read_csv(player_file,parse_dates = ['DATE'])

#read dfs file
dfs_file = raw_data_dir + 'DFS1819.csv'
dfs = pd.read_csv(dfs_file)

# clean data
df = load_clean(player,dfs)

#sample for testing pipeline
#df = df.sample(5000)
df.to_pickle(interim_data_dir+'clean_merge.pkl')

#build starting features and transformations
df = build_features(df)
df.to_pickle(interim_data_dir+ 'df_features.pkl')

#make player and defense clusters
X_players, center_players, player_df = kmean_players_df(df,num_clusters=5)
X_defenses, center_defenses, defense_df = kmean_defenses_df(df,num_clusters=5)

#merge player and defense clusters onto main df
df = pd.merge(df,player_df,how = 'left',on = ['player_id','date'])
df = pd.merge(df,defense_df,how = 'left',on = ['opp','date'])

#lag everything
ncols = df.iloc[:,df.columns.get_loc('min'):].columns
df[[f"{i}_lag" for i in ncols]] = df.groupby('player_id') \
                                [ncols].transform(lambda x : x.shift())


#create salary change 
df['sal_prev']= df['salary']-df['salary_lag']

#final data frame processing
df.dropna(inplace=True)
df.sort_values(by='date',inplace=True)
df.reset_index(inplace=True)

#send to pickle
df.to_pickle(processed_data_dir+'full.pkl')




