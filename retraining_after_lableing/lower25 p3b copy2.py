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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from retraining_after_lableing.params import param_grid_p3b as param_grid
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

random_state = 41

erp_name = 'both'
targets = ['miscatch_P3a_novel', 'miscatch_P3b_target']
input_path = r"S:\Data_Science\Core\FDA_submission_10_2019\08-Reports\STAR_reports\Labeling_project\kaggle_miscatches\both"


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

# splitting to agebins should be done here
df = pd.merge(y_train, X_train, on='taskData._id.$oid')

df = df[(df['agebin'] == "aob_12-16F") | (df['agebin'] == "aob_12-16M") | (df['agebin'] == "aob_14-19F") | (df['agebin'] == "aob_14-19M")]

dq_df = pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Target.csv")
features_p3b = [i for i in dq_df.columns if ('P3b' in i) and ('miscatch' not in i)]


dq_df = dq_df[features_p3b+['taskData.elm_id']].merge(pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Novel.csv"), on='taskData.elm_id')
dq_df['agebin'] = dq_df.apply(get_agebin, axis=1)
features_p3a = [i for i in dq_df.columns if ('P3a' in i) and ('miscatch' not in i)]
features = features_p3a + features_p3b

new_p3a = pd.read_csv(r"C:\Users\nogag\aob-miscatch-detection\retraining_after_lableing\data\complete_lower25.csv")
dq_df = dq_df[dq_df['taskData.elm_id'].isin(new_p3a['taskData._id.$oid'])]
dq_df = dq_df[features+['taskData.elm_id', 'visit']].merge(new_p3a[['taskData._id.$oid', 'miscatch_P3b_target']], left_on='taskData.elm_id', right_on='taskData._id.$oid')
dq_df['miscatch_P3b_target'] = dq_df['miscatch_P3b_target'].apply(lambda x: 1 if x=='yes' else 0)

df = df.append(dq_df)

dq_df = dq_df.dropna(subset=features)
dq_df = dq_df[dq_df.visit==1]
df = df.dropna(subset=features)
df = df[df.visit == 1]

## remove DQ list
dq = pd.read_csv(r"C:\Users\nogag\aob-miscatch-detection\DQ\AOB_Target_remove.csv")
df = df[~df['taskData._id.$oid'].isin(dq['taskData.elm_id'])]


pipeline = Pipeline(steps=[
        ("preprocess", StandardScaler()),
        ("feature_selection", RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear'))),
        ("estimator", CatBoostClassifier(verbose=0))
    ])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=StratifiedKFold(10), scoring='f1')
grid_search.fit(df[features], df["miscatch_P3b_target"], estimator__early_stopping_rounds=5)
clf = grid_search.best_estimator_
predictions = cross_val_predict(clf, df[features], df["miscatch_P3b_target"], cv=StratifiedKFold(10), fit_params={"estimator__early_stopping_rounds":5})
print(grid_search.best_params_)
print('auc',  roc_auc_score(df["miscatch_P3b_target"], predictions))
print('recall', recall_score(df["miscatch_P3b_target"], predictions))
print('precision', precision_score(df["miscatch_P3b_target"], predictions))
print('confusion martix\n', confusion_matrix(df["miscatch_P3b_target"], predictions), '\n')

df[predictions==1]['taskData._id.$oid'].to_csv(fr'C:\Users\nogag\aob-miscatch-detection\retraining_after_lableing\exports_p3b - Copy\pred_miscatches_lower25_cv_gridsearch.csv', index=False, header=['taskData.elm_id'])
df[(predictions == 1) & (df["miscatch_P3b_target"]==0)]['taskData._id.$oid'].to_csv(
        fr'C:\Users\nogag\aob-miscatch-detection\retraining_after_lableing\exports_p3b - Copy\false positive_miscatches_lower25_cv_gridsearch.csv',
        index=False, header=['taskData.elm_id'])
df[(predictions == 0) & (df["miscatch_P3b_target"]==1)]['taskData._id.$oid'].to_csv(
        fr'C:\Users\nogag\aob-miscatch-detection\retraining_after_lableing\exports_p3b - Copy\false negative_miscatches_lower25_cv_gridsearch.csv',
        index=False, header=['taskData.elm_id'])


all_data = pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Target.csv")

all_data = all_data[features_p3b+['taskData.elm_id']].merge(pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Novel.csv"), on='taskData.elm_id')
all_data['agebin'] = all_data.apply(get_agebin, axis=1)
all_data = all_data[(all_data['agebin'] == "aob_12-16F") | (all_data['agebin'] == "aob_12-16M") | (all_data['agebin'] == "aob_14-19F") | (all_data['agebin'] == "aob_14-19M")]
df = df.dropna(subset=features)
all_data = all_data.dropna(subset=features)
all_data = all_data[all_data.visit==1]
all_data = all_data[~all_data['taskData.elm_id'].isin(dq['taskData.elm_id'])]

X_pred = all_data[~all_data['taskData.elm_id'].isin(df['taskData._id.$oid'])]
grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='f1', cv=StratifiedKFold(10))
grid_search.fit(df[features], df["miscatch_P3b_target"], estimator__early_stopping_rounds=5)
y_pred = grid_search.best_estimator_.fit(df[features], df["miscatch_P3b_target"]).predict(X_pred)
print(grid_search.best_params_)


X_pred[y_pred==1]['taskData.elm_id'].to_csv(fr'C:\Users\nogag\aob-miscatch-detection\retraining_after_lableing\exports_p3b - Copy\pred_miscatches_lower25_unlabeleddata.csv', index=False, header=['taskData.elm_id'])
