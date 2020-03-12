from imblearn.over_sampling import SMOTE
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix
import os
from scipy.stats import iqr, zscore
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest, AdaBoostClassifier
from imblearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from retraining_after_lableing.params import param_grid

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



def reference_get_agebins(x):
    reference_ages = []
    if (x.gender == 'Female') & (x.ageV1 <= 16):
        reference_ages.append("aob_12-16F")

    if (x.gender == 'Male') & (x.ageV1 <= 16):
        reference_ages.append("aob_12-16M")

    if (x.gender == 'Female') & (x.ageV1 > 14) & (x.ageV1 <= 19):
        reference_ages.append("aob_14-19F")

    if (x.gender == 'Male') & (x.ageV1 > 14) & (x.ageV1 <= 19):
        reference_ages.append("aob_14-19M")

    if (x.ageV1 > 18) & (x.ageV1 <= 25):
        reference_ages.append("aob_18-25")

    if (x.ageV1 > 25) & (x.ageV1 <= 39):
        reference_ages.append("aob_25-39")

    if (x.ageV1 > 35) & (x.ageV1 <= 50):
        reference_ages.append("aob_35-50")

    if (x.ageV1 > 50) & (x.ageV1 <= 65):
        reference_ages.append("aob_50-65")

    if (x.ageV1 > 65) & (x.ageV1 <= 75):
        reference_ages.append("aob_65-75")

    if x.ageV1 > 75:
        reference_ages.append("aob_75-85")

    return str(reference_ages).strip('[]')



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


# splitting to agebins should be done here
df = pd.merge(y_train, X_train, on='taskData._id.$oid')
df = df[(df['agebin'] == "aob_50-65") | (df['agebin'] == "aob_65-75")| (df['agebin'] == "aob_75-85")]

dq_df = pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Target.csv")
new_p3a = pd.read_csv(r"C:\Users\nogag\aob-miscatch-detection\retraining_after_lableing\data\complete_higher50.csv")
dq_df = dq_df[dq_df['taskData.elm_id'].isin(new_p3a['taskData._id.$oid'])]
dq_df = dq_df[features+['taskData.elm_id', 'visit']].merge(new_p3a[['taskData._id.$oid', 'miscatch_P3b_target']], left_on='taskData.elm_id', right_on='taskData._id.$oid')
dq_df['miscatch_P3b_target'] = dq_df['miscatch_P3b_target'].apply(lambda x: 1 if x=='yes'  else 0 )
df = df.append(dq_df)
cv = StratifiedKFold(3)
dq_df = dq_df.dropna(subset=features)
df = df.dropna(subset=features)
df = df[df.visit == 1]
i =0

for train_index, test_index in cv.split(df[features], df['miscatch_P3b_target']):
    df_train, df_test = df.iloc[train_index], df.iloc[test_index]

    Xt = df_train[features]
    Xv = df_test[features]
    yt = df_train['miscatch_P3b_target']
    yv = df_test['miscatch_P3b_target']
    export_ids = df_test['taskData._id.$oid']

    pipeline = Pipeline(steps=[
            ("preprocess", StandardScaler()),
        ("sampling", SMOTE()),
        ("feature_selection", RFE(LogisticRegression(solver='liblinear'))),
            ("estimator", CatBoostClassifier(verbose=0))
        ])

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='roc_auc', cv=StratifiedKFold(3))
    grid_search.fit(Xt, yt)
    target_out = grid_search.predict(Xv)

    print(grid_search.best_params_)
    print('auc',  roc_auc_score(yv, target_out))
    print('recall', recall_score(yv, target_out))
    print('precision', precision_score(yv, target_out))
    print('confusion martix\n', confusion_matrix(yv, target_out), '\n')

    export_ids[target_out==1].to_csv(fr'C:\Users\nogag\aob-miscatch-detection\retraining_after_lableing\exports_p3b\pred_miscatches_higher50_cv_{i}.csv', index=False, header=['taskData.elm_id'])

    i = i+1



all_data = pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Target.csv")

all_data['agebin'] = all_data.apply(get_agebin, axis=1)
all_data = all_data[(all_data['agebin'] == "aob_50-65") | (all_data['agebin'] == "aob_65-75") | (all_data['agebin'] == "aob_75-85")]
df = df.dropna(subset=features)
all_data = all_data.dropna(subset=features)
X_pred = all_data[~all_data['taskData.elm_id'].isin(df['taskData._id.$oid'])]

pipeline = Pipeline(steps=[
    ("preprocess", StandardScaler()),
    ("sampling", SMOTE()),
    ("feature_selection", RFE(LogisticRegression(solver='liblinear'))),
    ("estimator", CatBoostClassifier(verbose=0))
])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='roc_auc', cv=StratifiedKFold(3))
grid_search.fit(df[features], df['miscatch_P3b_target'])
target_out = grid_search.predict(X_pred[features])
X_pred[target_out==1]['taskData.elm_id'].to_csv(fr'C:\Users\nogag\aob-miscatch-detection\retraining_after_lableing\exports_p3b\pred_miscatches_higher50_unlabeleddata.csv', index=False, header=['taskData.elm_id'])