import pandas as pd
import os
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)
from datetime import date

def load_clean(player,dfs):
       #change player column names
       player.columns = ['DATASET', 'GAME_ID', 'DATE', 'PLAYER_ID', 'NAME', \
              'POSITION', 'TEAM', 'OPP', 'VENUE', \
              'START', 'MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'OR', \
              'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS', 'USAGE', \
              'REST']

       #clean dfs file
       keep_cols = [1,3,13,16]
       dfs  = dfs.iloc[:,keep_cols].copy()
       dfs.columns = dfs.iloc[0,:]
       dfs.drop([0],axis=0,inplace=True)
       dfs.columns = ['GAME_ID','PLAYER_ID','POS','SALARY']
       dfs['GAME_ID'] = dfs.GAME_ID.astype(int)
       dfs['PLAYER_ID'] = dfs.PLAYER_ID.astype(int)

       #merge player and dfs files
       df = player.merge(dfs,how = 'left',on = ['GAME_ID','PLAYER_ID'])

       #clean merged file
       df.dropna(inplace=True)
       df.reset_index(drop=True,inplace=True)
       df.columns  = [i.lower() for i in df.columns]
       df.drop(['position'],axis=1,inplace=True)
       df['salary'] = df.salary.astype(int)

       #reorder columns
       cols = list(df.columns.values) 
       cols.pop(cols.index('rest')) 
       cols.pop(cols.index('pos')) 
       df = df[['rest','pos']+ cols] 

       return df


if __name__ == "__main__":
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

       load_clean(player,dfs)

       df.to_pickle(interim_data_dir+'clean_merge.pkl')