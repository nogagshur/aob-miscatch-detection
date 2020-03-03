from keras.optimizers import Adam
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import os
from scipy.stats import iqr, zscore
from shutil import copyfile
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.ensemble import IsolationForest, AdaBoostClassifier
from sklearn.svm import OneClassSVM

from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from catboost import CatBoostClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols

random_state = 41

#erp_name = 'miscatch_P3b_target'
erp_name = 'miscatch_P3a_novel'
targets = ['miscatch_P3a_novel', 'miscatch_P3b_target']
input_path = rf"S:\Data_Science\Core\FDA_submission_10_2019\08-Reports\STAR_reports\Labeling_project\kaggle_miscatches\{erp_name}"


def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


class MahalanobisOneclassClassifier():
    def __init__(self, xtrain, significance_level=0.01):
        self.xtrain = xtrain
        self.critical_value = chi2.ppf((1 - significance_level), df=xtrain.shape[1] - 1)

    def predict_proba(self, xtest):
        mahalanobis_dist = mahalanobis(xtest, self.xtrain)
        self.pvalues = 1 - chi2.cdf(mahalanobis_dist, 2)
        return mahalanobis_dist

def make_unsup(df, all_data, features):
    moc = MahalanobisOneclassClassifier(all_data[features])
    gmm2 = GaussianMixture(n_components=2).fit(all_data[features])
    gmm3 = GaussianMixture(n_components=3).fit(all_data[features])
    gmm4 = GaussianMixture(n_components=4).fit(all_data[features])
    osvm = OneClassSVM().fit(all_data[features])
    isof = IsolationForest().fit(all_data[features])

    all_data['moc'] = moc.predict_proba(all_data[features])
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

def get_agebin(x):
    if (x.gender == 'Female') & (x.ageV1 <= 15.39):
        return "aob_12-16F"

    if (x.gender == 'Male') & (x.ageV1 <= 14.8):
        return "aob_12-16M"

    if (x.gender == 'Female') & (x.ageV1 > 15.39) & (x.ageV1 <= 18.63):
        return "aob_14-19F"

    if (x.gender == 'Male') & (x.ageV1 > 14.8) & (x.ageV1 <= 18.63):
        return "aob_14-19M"

    if (x.ageV1 > 18.63) & (x.ageV1 <= 25):
        return "aob_18-25"

    if (x.ageV1 > 25) & (x.ageV1 <= 37.19):
        return "aob_25-39"

    if (x.ageV1 > 37.19) & (x.ageV1 <= 50):
        return "aob_35-50"

    if (x.ageV1 > 50) & (x.ageV1 <= 65):
        return "aob_50-65"

    if (x.ageV1 > 65) & (x.ageV1 <= 75):
        return "aob_65-75"

    if x.ageV1 > 75:
        return "aob_75-85"


X_train = pd.read_csv(os.path.join(input_path, "X_train.csv"))
y_train = pd.read_csv(os.path.join(input_path, "y_train.csv"))
X_test = pd.read_csv(os.path.join(input_path, "X_test.csv"))
y_test = pd.read_csv(os.path.join(input_path, "y_test.csv"))



features = ['P3b_Delta_Target_similarity_spatial',
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

features = [
             'P3b_Delta_Target_attr_timeMSfromTriger',
             'P3b_Delta_Target_attr_leftRight',
             'P3b_Delta_Target_attr_posteriorAnterior',
             'P3b_Delta_Target_attr_amplitude']


# features = ['P3a_Delta_Novel_attr_timeMSfromTriger',
#             'P3a_Delta_Novel_attr_leftRight',
#             'P3a_Delta_Novel_attr_posteriorAnterior',
#             'P3a_Delta_Novel_attr_amplitude']


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
            'P3a_Delta_Novel_topo_topographicSimilarity']

# splitting to agebins should be done here

df_train = pd.merge(y_train, X_train, on='taskData._id.$oid')
df_test = pd.merge(y_test, X_test, on='taskData._id.$oid')
df = pd.concat([df_train, df_test])
#dq_df = pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Target.csv")
dq_df = pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Novel.csv")
dq_df['agebin'] = dq_df.apply(get_agebin, axis=1)

dq_df = dq_df.dropna(subset=features)
df = df.dropna(subset=features)

ss = StandardScaler()
all_data = pd.DataFrame(ss.fit_transform(dq_df[features]), columns=features)
df[features] = ss.transform(df[features])

all_data['taskData._id.$oid'] = dq_df['taskData.elm_id'].values
all_data['agebin'] = dq_df['agebin'].values
agebins_df = []
for age in df['agebin'].unique():
    agedf = df[df['agebin'] == age]
    agedf = make_unsup(agedf, all_data[all_data['agebin'] == age], features)
    # IQR & Z score
    agebins_df.append(agedf)

features = ['moc', 'osvm', 'isof', 'gmm2_0', 'gmm2_1', 'gmm3_0', 'gmm3_1', 'gmm3_2', 'gmm4_0', 'gmm4_1', 'gmm4_2','gmm4_3', 'iqr', 'zscore', 'taskData._id.$oid']
df = pd.concat(agebins_df).dropna(subset=features)[features]
#df.to_csv(r'S:\Data_Science\Core\FDA_submission_10_2019\08-Reports\STAR_reports\Labeling_project\DQ\unsupervised\P3b\unsupervised_attr_only.csv')
df.to_csv(r'S:\Data_Science\Core\FDA_submission_10_2019\08-Reports\STAR_reports\Labeling_project\DQ\unsupervised\P3a\unsupervised.csv')
