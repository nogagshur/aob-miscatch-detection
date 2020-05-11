import os
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix

from imblearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from catboost import CatBoostClassifier
from retraining_after_lableing.params import param_grid_p3a

from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
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

missing_in_Eran_pool = ["591480fc3506906168f8a83f", "56767908fcc5549d1b32f518", "55f360b132991e86075a22a8"]

y_train = pd.read_csv(os.path.join(input_path, "y_train.csv"))
y_test = pd.read_csv(os.path.join(input_path, "y_test.csv"))
new_labels = pd.read_csv(r"C:\Users\nogag\aob-miscatch-detection\retraining_after_lableing\data\complete_lower25.csv")[['miscatch_P3a_novel','miscatch_P3b_target', 'taskData._id.$oid']]
labels_data = pd.concat([y_train, y_test, new_labels])
labels_data = labels_data.drop_duplicates("taskData._id.$oid")
labels_data = labels_data[~labels_data["taskData._id.$oid"].isin(missing_in_Eran_pool)]
labels_data['miscatch_P3b_target'] = labels_data['miscatch_P3b_target'].apply(lambda x: 1 if x == 'yes' else 0)
labels_data['miscatch_P3a_novel'] = labels_data['miscatch_P3a_novel'].apply(lambda x: 1 if x == 'yes' else 0)

target_features = pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Target.csv")
features_p3b = [i for i in target_features.columns if ('P3b' in i)]

miscatches_features = target_features[features_p3b+['taskData.elm_id']].merge(pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Novel.csv"), on='taskData.elm_id')
features_p3a = [i for i in miscatches_features.columns if ('P3a' in i) ]

features = features_p3a + features_p3b
miscatches_features['agebin'] = miscatches_features.apply(get_agebin, axis=1)
miscatches_features['reference_agebin'] = miscatches_features.apply(reference_get_agebins, axis=1)

miscatches_features = miscatches_features.drop_duplicates("taskData.elm_id")

miscatches_features = miscatches_features[features+['taskData.elm_id', 'visit', 'agebin', 'reference_agebin']].merge(labels_data[['taskData._id.$oid', 'miscatch_P3b_target', 'miscatch_P3a_novel']], left_on='taskData.elm_id', right_on='taskData._id.$oid', how="left")


df = miscatches_features[(miscatches_features['agebin'] == "aob_12-16F") | (miscatches_features['agebin'] == "aob_12-16M") | (miscatches_features['agebin'] == "aob_14-19F") | (miscatches_features['agebin'] == "aob_14-19M")]# | (miscatches_features['agebin'] == "aob_18-25")]
df = df.dropna(subset=features)
df = df[df.visit == 1]

remove_DQ = []
if "target" in remove_DQ:
    dq = pd.read_csv(r"C:\Users\nogag\aob-miscatch-detection\DQ\AOB_Target_remove.csv")
    df = df[~df['taskData._id.$oid'].isin(dq['taskData.elm_id'])]
if "novel" in remove_DQ:
    dq = pd.read_csv(r"C:\Users\nogag\aob-miscatch-detection\DQ\AOB_Novel_remove.csv")
    df = df[~df['taskData._id.$oid'].isin(dq['taskData.elm_id'])]
df.to_csv(r"C:\Users\nogag\aob-miscatch-detection\data\non_dq_data.csv")

df = df.dropna(subset=features+["miscatch_P3a_novel", "miscatch_P3b_target"])

pipeline = Pipeline(steps=[
        ("preprocess", StandardScaler()),
        ("feature_selection", RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear'))),
        ("estimator", CatBoostClassifier(verbose=0))
    ])


features2 = ['P3a_Delta_Novel_similarity_spatial',
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

df['miscatches'] = (df["miscatch_P3a_novel"] + df["miscatch_P3b_target"]) % 2

for tmp_features in [features2]:
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
