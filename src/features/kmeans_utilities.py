import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score


def kmean_players_df(df, num_clusters =5):
    X = df.copy()
    X = X[['pts_cum_avg','tot_cum_avg','a_cum_avg','st_cum_avg','bl_cum_avg']]

    std = StandardScaler()
    std.fit(X)
    X_scaled = std.transform(X)

    km = KMeans(n_clusters=num_clusters,random_state=10)
    km.fit(X)

    centers = pd.DataFrame(km.cluster_centers_)
    centers.columns = X.columns

    X['name'] = df.name
    X['player_type'] = km.labels_
    X['date'] = df.date 
    X['value'] = df.value 
    X['player_id'] = df.player_id

    X.head()
    player_types = X[['player_id','date','player_type']]

    return X_scaled,centers,player_types



def kmean_defenses_df(df,num_clusters=5):
    X = df.copy()
    X = X.groupby(['opp','date']) \
                        ['fp_fg','fp_3p','fp_ft','fp_or','fp_dr','fp_tot','fp_a','fp_st','fp_bl'].sum()

    X.reset_index(inplace=True)

    ncols = X.iloc[:,X.columns.get_loc('fp_fg'):].columns
    X[[f"{i}_total_cum_avg" for i in ncols]] = X.groupby('opp') \
                                            [ncols].transform(lambda x : x.expanding(axis=0).mean())
    

    X_def_kmeans = X.iloc[:,11:]

    std = StandardScaler()
    std.fit(X_def_kmeans)
    X_def_kmeans_scaled = std.transform(X_def_kmeans)

    num_clusters=num_clusters
    km = KMeans(n_clusters=num_clusters,random_state=10)
    km.fit(X_def_kmeans_scaled)

    centers = pd.DataFrame(km.cluster_centers_)
    centers.columns = X_def_kmeans.columns

    X_def_kmeans['opp'] = X.opp
    X_def_kmeans['def_type'] = km.labels_
    X_def_kmeans['date'] = X.date

    def_types = X_def_kmeans[['opp','date','def_type']]

    return X_def_kmeans_scaled,centers,def_types



def kmean_inertia_validate(X,n):
    inertia = []
    clusters = list(range(2,n))

    for clust in clusters:
        km = KMeans(n_clusters = clust)
        km.fit(X)
        inertia.append(km.inertia_)
    
    plt.plot(clusters,inertia)
    plt.scatter(clusters,inertia)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Player Clusters - Inertia Validation')
    plt.show()


def kmean_silhouette_validate(X,n):
    silh = []
    clusters = list(range(2,n))

    for clust in clusters:
        km = KMeans(n_clusters=clust, random_state=10)
        labels = km.fit_predict(X)
        silhouette_avg = silhouette_score(X,labels)
        silh.append(silhouette_avg)
        
        
    plt.plot(clusters,silh)
    plt.scatter(clusters,silh)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score Average')
    plt.title('Player Clusters - Sihouette Validation')
    plt.show()


if __name__ == "__main__":
    raw_data_dir = '~/nba_dfs_value/data/raw/'
    interim_data_dir= '~/nba_dfs_value/data/interim/'
    processed_data_dir = '~/nba_dfs_value/data/processed/'
    ndf = pd.read_pickle(interim_data_dir+"df_features.pkl")

    
