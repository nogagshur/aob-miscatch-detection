from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.svm import OneClassSVM
import pandas as pd
from sklearn.pipeline import Pipeline
from statsmodels.formula.api import ols

from retraining_after_lableing.params import param_grid_p3a
pipeline = Pipeline(steps=[
        ("feature_selection", RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear'))),
        ("estimator", CatBoostClassifier(verbose=0))
    ])

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
def make_unsup(df, all_data, features):
    ss = StandardScaler().fit_transform(all_data[features])
    gmm2 = GaussianMixture(n_components=2).fit(ss)
    gmm3 = GaussianMixture(n_components=3).fit(ss)
    gmm4 = GaussianMixture(n_components=4).fit(ss)
    osvm = OneClassSVM().fit(ss)
    isof = IsolationForest().fit(all_data[features])
    all_data['osvm'] = osvm.predict(ss)
    all_data['isof'] = isof.predict(ss)
    all_data['gmm2_0'] = gmm2.predict_proba(ss)[:, 0]
    all_data['gmm2_1'] = gmm2.predict_proba(ss)[:, 1]
    all_data['gmm3_0'] = gmm3.predict_proba(ss)[:, 0]
    all_data['gmm3_1'] = gmm3.predict_proba(ss)[:, 1]
    all_data['gmm3_2'] = gmm3.predict_proba(ss)[:, 2]
    all_data['gmm4_0'] = gmm4.predict_proba(ss)[:, 0]
    all_data['gmm4_1'] = gmm4.predict_proba(ss)[:, 1]
    all_data['gmm4_2'] = gmm4.predict_proba(ss)[:, 2]
    all_data['gmm4_3'] = gmm4.predict_proba(ss)[:, 3]
   # all_data['iqr'] = iqr(all_data[features], axis=1)
   # all_data['zscore'] = zscore(all_data[features], axis=1).mean(axis=1)
    df = pd.merge(all_data[['osvm', 'isof', 'gmm2_0', 'gmm2_1', 'gmm3_0', 'gmm3_1', 'gmm3_2',
                            'gmm4_0', 'gmm4_1', 'gmm4_2','gmm4_3', 'taskData._id.$oid']], df, on="taskData._id.$oid", how='right')

    return df


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


df = pd.read_csv(r"C:\Users\nogag\aob-miscatch-detection\data\non_dq_data.csv")
df = df.dropna(subset=features+["miscatch_P3a_novel", "miscatch_P3b_target"])
df['miscatches'] = (df["miscatch_P3a_novel"] + df["miscatch_P3b_target"]) % 2
df = df.dropna(subset=features)

agebins_df = []
for age in df['agebin'].unique():
    agedf = df[df['agebin'] == age]
    agedf = make_unsup(agedf, df[df['reference_agebin'].str.contains(age)], features)

    agedf = cooks(agedf, df[df['reference_agebin'].str.contains(age)], 'P3a_Delta_Novel_matchScore', ['P3a_Delta_Novel_similarity_locationPA', 'P3a_Delta_Novel_similarity_timing',
            'P3a_Delta_Novel_similarity_amplitude',], 0)
    agedf = cooks(agedf, df[df['reference_agebin'].str.contains(age)], 'P3a_Delta_Novel_similarity_locationPA', ['P3a_Delta_Novel_attr_timeMSfromTriger',
            'P3a_Delta_Novel_similarity_amplitude'], 1)
    agedf = cooks(agedf, df[df['reference_agebin'].str.contains(age)], 'P3a_Delta_Novel_attr_posteriorAnterior',
                  ['P3a_Delta_Novel_attr_timeMSfromTriger',
    'P3a_Delta_Novel_attr_leftRight','P3a_Delta_Novel_attr_amplitude'], 2)
    agedf = cooks(agedf, df[df['reference_agebin'].str.contains(age)], 'P3a_Delta_Novel_matchScore',
                  ['P3a_Delta_Novel_attr_timeMSfromTriger', 'P3a_Delta_Novel_attr_posteriorAnterior',
    'P3a_Delta_Novel_attr_leftRight','P3a_Delta_Novel_attr_amplitude'], 3)
    agedf = cooks(agedf, df[df['reference_agebin'].str.contains(age)], 'P3a_Delta_Novel_similarity_timing', ['P3a_Delta_Novel_attr_timeMSfromTriger', 'P3a_Delta_Novel_attr_amplitude'],4)
    agedf = cooks(agedf,df[df['reference_agebin'].str.contains(age)], 'P3a_Delta_Novel_topo_topographicSimilarity', ['P3a_Delta_Novel_attr_timeMSfromTriger',
            'P3a_Delta_Novel_similarity_timing'], 5)
    agedf = cooks(agedf, df[df['reference_agebin'].str.contains(age)], 'P3a_Delta_Novel_topo_topographicCorrCoeffAligned',
                  ['P3a_Delta_Novel_similarity_locationLR', 'P3a_Delta_Novel_similarity_locationPA'], 6)

    agedf = cooks(agedf, df[df['reference_agebin'].str.contains(age)], 'P3a_Delta_Novel_matchScore',
                  ['P3a_Delta_Novel_topo_topographicCorrCoeffAligned',
                'P3a_Delta_Novel_topo_topographicSimilarity', 'P3a_Delta_Novel_similarity_spatial'], 7)
    agebins_df.append(agedf)

features = features +[ 'osvm', 'isof', 'gmm2_0', 'gmm2_1', 'gmm3_0', 'gmm3_1', 'gmm3_2', 'gmm4_0', 'gmm4_1', 'gmm4_2','gmm4_3',  'cooks_d0', 'cooks_d1','cooks_d2','cooks_d3', 'cooks_d4', 'cooks_d5','cooks_d6','cooks_d7']

df = pd.concat(agebins_df).dropna(subset=features)

for tmp_features in [features]:
    for es in [1, 7, 15]:
        for k in [4, 8]:
            for target in ["miscatch_P3a_novel", "miscatch_P3b_target", "miscatches"]:
                print("feature_set =", len(tmp_features))
                print("early stopping =", es)
                print("cv k =", k)
                print("target variable =", target)
                grid_search = GridSearchCV(pipeline, param_grid=param_grid_p3a, scoring='f1', cv=StratifiedKFold(k))
                grid_search.fit(df[tmp_features], df[target], estimator__early_stopping_rounds=es)
                clf = grid_search.best_estimator_
                predictions = cross_val_predict(clf, df[tmp_features], df[target], cv=StratifiedKFold(k), fit_params={"estimator__early_stopping_rounds":es})
                print(grid_search.best_params_)
                print('auc',  roc_auc_score(df[target], predictions))
                print('recall', recall_score(df[target], predictions))
                print('precision', precision_score(df[target], predictions))
                print('confusion martix\n', confusion_matrix(df[target], predictions), '\n')
