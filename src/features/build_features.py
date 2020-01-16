import pandas as pd
import numpy as np


def build_features(ndf):
    #create FP and value columns 

    ndf['fp_pts'] = ndf['pts']*1
    ndf['fp_fg'] = ndf['fg']*2
    ndf['fp_3p'] = ndf['3p']*3
    ndf['fp_ft'] = ndf['ft']*1
    ndf['fp_or'] = ndf['or']*1.2
    ndf['fp_dr'] = ndf['dr']*1.2
    ndf['fp_tot'] = ndf['tot']*1.2
    ndf['fp_a'] = ndf['a']*1.5
    ndf['fp_st'] = ndf['st']*3
    ndf['fp_to'] = ndf['to']*-1
    ndf['fp_bl'] = ndf['bl']*3
    ndf['fp'] = ndf.fp_pts+ndf.fp_tot+ndf.fp_a+ndf.fp_st+ndf.fp_to+ndf.fp_bl
    ndf['fpd'] = ndf.fp*1000/ndf.salary
    ndf['value'] = np.where(ndf['fpd']>5, 1, 0)
    ndf['rest'] = np.where(ndf['rest']=='3+',99,ndf['rest'])
    ndf['rest'] = pd.to_numeric(ndf['rest'])
    ndf['rest'] = np.where(ndf['rest']>3,'3+',ndf['rest'])

    #rearrange columns 
    ndf = ndf[['rest', 'pos', 'dataset', 'game_id', 'date', 'player_id', 'name',
       'team', 'opp', 'venue', 'start','fpd', 'value', 'min', 'fg', 'fga', '3p', '3pa', 'ft',
       'fta', 'or', 'dr', 'tot', 'a', 'pf', 'st', 'to', 'bl', 'pts', 'usage',
       'salary', 'fp_pts', 'fp_fg', 'fp_3p', 'fp_ft', 'fp_or', 'fp_dr',
       'fp_tot', 'fp_a', 'fp_st', 'fp_to', 'fp_bl', 'fp']]


    
    #set up per min columns
    ncols = ndf.iloc[:,ndf.columns.get_loc('min')+1:].columns
    ndf[[f"{i}_per_min" for i in ncols]] = ndf[ncols].transform(lambda x : x/ndf['min'])

    #set up fp% columns
    ncols = ndf.iloc[:,ndf.columns.get_loc('fp_pts'):ndf.columns.get_loc('fp')].columns
    ndf[[f"{i}_%_fp" for i in ncols]] = ndf[ncols].transform(lambda x : x/ndf['fp'])


    #create moving averages and cumulate average
    ncols = ndf.iloc[:,ndf.columns.get_loc('min'):ndf.columns.get_loc('fp_bl_%_fp')+1].columns
    ndf[[f"{i}_3MA" for i in ncols]] = ndf.groupby('player_id') \
                            [ncols].transform(lambda x : x.rolling(window=3,min_periods=1).mean())
    ndf[[f"{i}_5MA" for i in ncols]] = ndf.groupby('player_id') \
                            [ncols].transform(lambda x : x.rolling(window=5,min_periods=1).mean())
    ndf[[f"{i}_7MA" for i in ncols]] = ndf.groupby('player_id') \
                            [ncols].transform(lambda x : x.rolling(window=7,min_periods=1).mean())
    ndf[[f"{i}_10MA" for i in ncols]] = ndf.groupby('player_id') \
                            [ncols].transform(lambda x : x.rolling(window=10,min_periods=1).mean())
    ndf[[f"{i}_cum_avg" for i in ncols]] = ndf.groupby('player_id') \
                            [ncols].transform(lambda x : x.expanding(axis=0).mean())

    #cumsum calcs
    ncols = ndf.iloc[:,ndf.columns.get_loc('min'):ndf.columns.get_loc('fp')+1].columns
    ndf[[f"{i}_cum_sum" for i in ncols]] = ndf.groupby('player_id') \
                        [ncols].transform(lambda x : x.expanding(axis=0).sum())


    #per min cumsum metrics
    ncols = ndf.iloc[:,ndf.columns.get_loc('fg_cum_sum'):ndf.columns.get_loc('fp_cum_sum')+1].columns
    ndf[[f"{i}_per_min" for i in ncols]] = ndf[ncols].transform(lambda x : x/ndf['min_cum_sum'])

    #per cum sum fp %
    ncols = ndf.iloc[:,ndf.columns.get_loc('fp_fg_cum_sum'):ndf.columns.get_loc('fp_cum_sum')].columns
    ndf[[f"{i}_%_fp_cum_sum" for i in ncols]] = ndf[ncols].transform(lambda x : x/ndf['fp_cum_sum'])


    #make dummy columns for position
    #ndf = pd.get_dummies(ndf,columns=['pos'])

    #find last start
    ndf['last_start'] = ndf.groupby('player_id') \
                ['start'].transform(lambda x : x.shift(1))


    ndf['new_start'] = np.where((ndf['start'] == 'Y')
                & (ndf['last_start'] =='N'), 
                1,     
                0) 

    return ndf


if __name__ == "__main__":
    raw_data_dir = '~/nba_dfs_value/data/raw/'
    interim_data_dir= '~/nba_dfs_value/data/interim/'
    processed_data_dir = '~/nba_dfs_value/data/processed/'
    ndf = pd.read_pickle(interim_data_dir+"clean_merge.pkl")

    ndf = build_features(ndf)
    ndf.to_pickle(interim_data_dir+ 'df_features.pkl')
