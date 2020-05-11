import itertools
import matplotlib.pyplot as plt

import os
from scipy.stats import iqr, zscore
import pandas as pd

from sklearn.ensemble import IsolationForest, AdaBoostClassifier
from sklearn.svm import OneClassSVM

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import pandas_profiling as pp
from catboost import CatBoostClassifier
from sklearn.mixture import GaussianMixture

def make_unsup(df, all_data, features):
    gmm2 = GaussianMixture(n_components=2).fit(all_data[features])
    gmm3 = GaussianMixture(n_components=3).fit(all_data[features])
    gmm4 = GaussianMixture(n_components=4).fit(all_data[features])
    osvm = OneClassSVM().fit(all_data[features])
    isof = IsolationForest().fit(all_data[features])

    all_data['osvm'] = osvm.predict(all_data[features])
    all_data['isof'] = isof.predict(all_data[features])
    all_data['gmm2_0'] = gmm2.predict_proba(all_data[features])[:, 0]
    all_data['gmm2_1'] = gmm2.predict_proba(all_data[features])[:, 1]
    all_data['gmm3_0'] = gmm3.predict_proba(all_data[features])[:, 0]
    all_data['gmm3_1'] = gmm3.predict_proba(all_data[features])[:, 1]
    all_data['gmm3_2'] = gmm3.predict_proba(all_data[features])[:, 2]
    all_data['gmm4_0'] = gmm4.predict_proba(all_data[features])[:, 0]
    all_data['gmm4_1'] = gmm4.predict_proba(all_data[features])[:, 1]
    all_data['gmm4_2'] = gmm4.predict_proba(all_data[features])[:, 2]
    all_data['gmm4_3'] = gmm4.predict_proba(all_data[features])[:, 3]
    all_data['iqr'] = iqr(all_data[features], axis=1)
    all_data['zscore'] = zscore(all_data[features], axis=1).mean(axis=1)
    df = pd.merge(all_data[['moc', 'osvm', 'isof', 'gmm2_0', 'gmm2_1', 'gmm3_0', 'gmm3_1', 'gmm3_2',
                            'gmm4_0', 'gmm4_1', 'gmm4_2','gmm4_3', 'iqr', 'zscore', 'taskData._id.$oid']], df, on="taskData._id.$oid", how='right')

    return df


pandas_profiling_output_path = r"C:\Users\nogag\aob-miscatch-detection\pandas_profiling"
def profile(df, folder, name):
    profile = pp.ProfileReport(df)

    if not os.path.exists(os.path.join(pandas_profiling_output_path,folder)):
        os.makedirs(os.path.join(pandas_profiling_output_path,folder))

    profile.to_file(os.path.join(pandas_profiling_output_path,folder, f'{name}_pandas_profile.html'))




features = ['P3a_Delta_Novel_similarity_spatial',
            'P3a_Delta_Novel_similarity_locationLR',
            'P3a_Delta_Novel_similarity_locationPA',
            'P3a_Delta_Novel_similarity_timing',
            'P3a_Delta_Novel_similarity_amplitude',
            'P3a_Delta_Novel_matchScore',
            'P3a_Delta_Novel_attr_timeMSfromTriger',
            'P3a_Delta_Novel_attr_leftRight',
            'P3a_Delta_Novel_attr_posteriorAnterior',
            'P3a_Delta_Novel_attr_amplitude',
            'P3a_Delta_Novel_topo_topographicCorrCoeffAligned',
            'P3a_Delta_Novel_topo_topographicSimilarity',
            'P3b_Delta_Target_similarity_spatial',
            'P3b_Delta_Target_similarity_locationLR',
            'P3b_Delta_Target_similarity_locationPA',
            'P3b_Delta_Target_similarity_timing',
            'P3b_Delta_Target_similarity_amplitude',
            'P3b_Delta_Target_matchScore',
            'P3b_Delta_Target_attr_timeMSfromTriger',
            'P3b_Delta_Target_attr_leftRight',
            'P3b_Delta_Target_attr_posteriorAnterior',
            'P3b_Delta_Target_attr_amplitude',
            'P3b_Delta_Target_topo_topographicCorrCoeffAligned',
            'P3b_Delta_Target_topo_topographicSimilarity']






if __name__ == "__main__":

    df = pd.read_csv(r"C:\Users\nogag\aob-miscatch-detection\data\non_dq_data.csv")
    df['miscatches'] = (df["miscatch_P3a_novel"] + df["miscatch_P3b_target"]) % 2

    for age in list(df['reference_agebin'].unique()):
        profile = pp.ProfileReport(df[(df["reference_agebin"] == age)])

        if not os.path.exists(os.path.join(pandas_profiling_output_path)):
            os.makedirs(os.path.join(pandas_profiling_output_path))
        profile.to_file(os.path.join(pandas_profiling_output_path, f'{age}.html'))

    if not os.path.exists(os.path.join(pandas_profiling_output_path)):
        os.makedirs(os.path.join(pandas_profiling_output_path))
    profile.to_file(os.path.join(pandas_profiling_output_path, f'all.html'))


    scatter_output_path = r"C:\Users\nogag\aob-miscatch-detection\EDA_scatter"

    for i,j in itertools.permutations(features, 2):
        name = f'{i} X {j}.jpg'
        plt.clf()
        plt.scatter(df[i], df[j], c='k', label='unlabeled')
        plt.scatter(df[df["miscatches"] == 0][i], df[df["miscatches"] == 0][j], c='b', label='non miscatch')
        plt.scatter(df[df["miscatches"] == 1][i], df[df["miscatches"] == 1][j], c='r', label='miscatch P3a')
        plt.scatter(df[df['miscatch_P3b_target'] == 1][i], df[df['miscatch_P3b_target'] == 1][j], c='purple', label='miscatch P3b')
        plt.xlabel(i)
        plt.ylabel(j)
        plt.legend()
        if not os.path.exists(os.path.join(scatter_output_path)):
            os.makedirs(os.path.join(scatter_output_path))
        plt.savefig(os.path.join(scatter_output_path, name))