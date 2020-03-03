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
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest, AdaBoostClassifier
from sklearn.svm import OneClassSVM
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.activations import relu

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas_profiling as pp
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from collections import Counter, defaultdict
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols

random_state = 41

erp_name = 'both'
targets = ['miscatch_P3a_novel', 'miscatch_P3b_target']
input_path = r"S:\Data_Science\Core\FDA_submission_10_2019\08-Reports\STAR_reports\Labeling_project\kaggle_miscatches\both"


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



def train_autoencoder(df, enc_dim):
    # encoding dimension set to half on input (12 -> 6).
    # Feel free to play with this

    # simple parameters (we can add a LOT more)
    input_dim = int(df.values.shape[-1])
    encoding_dim = int(enc_dim)

    input_layer = Input(shape=(input_dim,))
    encoded_l1 = Dense(int(np.mean([encoding_dim, input_dim])),
                       activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded_l1)
    decoded_l1 = Dense(int(np.mean([encoding_dim, input_dim])),
                       activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded_l1)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(df.values, df.values, verbose=0,
                    epochs=2000,# definitely overfitted, but we should add a validation set + early stopping
                    # batch_size=32,
                    batch_size=317, #usually, it's preferable to use factors of 2, but this is a small dataset.
                    shuffle=True)

    return encoder

def encode_unlabeled(encoder, df):
    encoded_features = pd.DataFrame(data=encoder.predict(df.values))
    return encoded_features.add_prefix('encoded_feat_')


def autoencoder(df, all_data, features, enc_dim):
    #ToDo :
    num_df = all_data[features] #all_data.select_dtypes('number')
    encoder = train_autoencoder(df=num_df, enc_dim=enc_dim)
    encoded = encode_unlabeled(encoder=encoder, df=df[features])
    return encoded

def cooks(df, all_data, t, v, i):

    m = ols(
        f'{t} ~ {" + ".join(v)}',
        all_data).fit()
    infl = m.get_influence()
    sm_fr = infl.summary_frame()
    # step 4 - feature engineering on test data
    # IQR -
    all_data[f'cooks_d{i}'] = sm_fr['cooks_d']
    all_data[f'dffits{i}'] = sm_fr['dffits']
    df = pd.merge(all_data[[f'cooks_d{i}', f'dffits{i}', 'taskData._id.$oid']], df, on="taskData._id.$oid", how='right')
    return df

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

features = [i for i in X_train.columns if ((('P3b' in i) or ('P3a' in i)) and ('miscatch' not in i))]

# splitting to agebins should be done here
df = pd.merge(y_train, X_train, on='taskData._id.$oid')

df = df[(df['agebin'] == "aob_50-65") | (df['agebin'] == "aob_65-75")| (df['agebin'] == "aob_75-85")]

dq_df = pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Novel.csv")
featuresP3a = [i for i in dq_df.columns if ('P3a' in i) and ('miscatch' not in i)] + ['taskData.elm_id']
dq_df = dq_df[featuresP3a].merge(pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Target.csv"), on='taskData.elm_id')
dq_df['agebin'] = dq_df.apply(get_agebin, axis=1)
dq_df = dq_df[(dq_df['agebin'] == "aob_50-65") | (dq_df['agebin'] == "aob_65-75") | (dq_df['agebin'] == "aob_75-85")]


cv = StratifiedKFold(7)
#all_d#ata = dq_df[features]
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

df = pd.concat(agebins_df).dropna(subset=features)
# #
features = features +['moc', 'osvm', 'isof', 'gmm2_0', 'gmm2_1', 'gmm3_0', 'gmm3_1', 'gmm3_2', 'gmm4_0', 'gmm4_1', 'gmm4_2','gmm4_3', 'iqr', 'zscore']
df = df.dropna(subset=features)
df['target'] = df[targets[0]] | df[targets[1]]
df_vis1 = df[df.visit == 1]
df_vis2 = df[df.visit == 2]

for train_index, test_index in cv.split(df_vis1[features+targets], df_vis1['target']):
    df_vis1_train, df_vis1_test = df_vis1.iloc[train_index], df_vis1.iloc[test_index]
    df_vis2_train = df_vis2[df_vis2['reference'].isin(list(df_vis1_train['reference']))]
    df_vis2_test = df_vis2[df_vis2['reference'].isin(list(df_vis1_test['reference']))]
    Xt = pd.concat([df_vis1_train[features], df_vis2_train[features]])
    Xv = pd.concat([df_vis1_test[features], df_vis2_test[features]])
    yt = pd.concat([df_vis1_train[targets + ['target']], df_vis2_train[targets + ['target']]])
    yv = pd.concat([df_vis1_test[targets + ['target']], df_vis2_test[targets + ['target']]])

    target_out0 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8).fit(Xt, yt[targets[0]], ).predict_proba(Xv)[:, 0]
    target_out1 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8).fit(Xt, yt[targets[1]]).predict_proba(Xv)[:, 0]
    target_out2 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8).fit(Xt, yt['target']).predict_proba(Xv)[:, 0]

    target_out = target_out0 * target_out1 * target_out2
    print('auc',  roc_auc_score(yv['target'], 1 - target_out))
    print('recall', recall_score(yv['target'], target_out < 0.75))
    print('precision', precision_score(yv['target'], target_out < 0.75))
    print('confusion martix\n', confusion_matrix(yv['target'], target_out < 0.75), '\n')


### test set


X_test = pd.read_csv(os.path.join(input_path, "X_test.csv"))
y_test = pd.read_csv(os.path.join(input_path, "y_test.csv"))

features = [i for i in X_train.columns if ((('P3b' in i) or ('P3a' in i)) and ('miscatch' not in i))]

# splitting to agebins should be done here
df_test = pd.merge(y_test, X_test, on='taskData._id.$oid')

df_test = df_test[(df_test['agebin'] == "aob_50-65") | (df_test['agebin'] == "aob_65-75")| (df_test['agebin'] == "aob_75-85")]

df_test = df_test.dropna(subset=features)

ss = StandardScaler()
all_data = pd.DataFrame(ss.fit_transform(dq_df[features]), columns=features)
df_test[features] = ss.transform(df_test[features])

all_data['taskData._id.$oid'] = dq_df['taskData.elm_id'].values
all_data['agebin'] = dq_df['agebin'].values
agebins_df = []
for age in df_test['agebin'].unique():
    agedf = df_test[df_test['agebin'] == age]
    agedf = make_unsup(agedf, all_data[all_data['agebin'] == age], features)
    # IQR & Z score
    agebins_df.append(agedf)

df_test = pd.concat(agebins_df).dropna(subset=features)
####

features = features +['moc', 'osvm', 'isof', 'gmm2_0', 'gmm2_1', 'gmm3_0', 'gmm3_1', 'gmm3_2', 'gmm4_0', 'gmm4_1', 'gmm4_2','gmm4_3', 'iqr', 'zscore']
df_test = df_test.dropna(subset=features)
df_test['target'] = df_test[targets[0]] | df_test[targets[1]]

target_out0 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8).fit(df[features], df[targets[0]]).predict_proba(df_test[features])[:, 0]
target_out1 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8).fit(df[features], df[targets[1]]).predict_proba(df_test[features])[:, 0]
target_out2 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8).fit(df[features], df['target']).predict_proba(df_test[features])[:, 0]

target_out = target_out0 * target_out1 * target_out2
print('auc',  roc_auc_score(df_test['target'], 1 - target_out))
print('recall', recall_score(df_test['target'], target_out < 0.75))
print('precision', precision_score(df_test['target'], target_out < 0.75))
print('confusion martix\n', confusion_matrix(df_test['target'], target_out < 0.75), '\n')

clf = CatBoostClassifier(verbose=0, class_weights=[1, 5]).fit(df[features], df['target'])
for i, j in sorted(zip(clf.feature_importances_, features), reverse=1):
    print(i, j)


### test set

df = pd.concat([df, df_test])
features = features +['moc', 'osvm', 'isof', 'gmm2_0', 'gmm2_1', 'gmm3_0', 'gmm3_1', 'gmm3_2', 'gmm4_0', 'gmm4_1', 'gmm4_2','gmm4_3', 'iqr', 'zscore']

X_pred = all_data[~all_data['taskData._id.$oid'].isin(df['taskData._id.$oid'])][features]


target_out0 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8).fit(df[features], df[targets[0]]).predict(X_pred)[:, 0]
target_out1 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8).fit(df[features], df[targets[1]]).predict(X_pred)[:, 0]
target_out2 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8).fit(df[features], df['target']).predict(X_pred)[:, 0]

target_out = target_out0 * target_out1 * target_out2
print(all_data[target_out<0.75]['taskData._id.$oid'])
all_data[target_out<0.75]['taskData._id.$oid'].to_csv('delegated_reports_higher50.csv')
