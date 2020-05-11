from imblearn.over_sampling import BorderlineSMOTE, SMOTE

from sklearn.naive_bayes import GaussianNB


from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import StandardScaler






param_grid_p3a = [
    ## done
    # {
    #     "preprocess": [PowerTransformer()],
    #     'estimator': [SVC()],
    #         "feature_selection": [SelectFromModel(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear',
    #                                                      class_weight='balanced', C=5))],
    #          "feature_selection__max_features": [ 6, 8, 10, 12, 14],
    #         'estimator__tol': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    #     'estimator__C': [1, 5,10,15, 20],
    #     'estimator__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    #     'estimator__class_weight': [{0:1, 1: 6}, {0:1, 1: 2}, {0:1, 1: 3}, {0:1, 1: 4}]},
    # # ## done
    {

          "estimator": [CatBoostClassifier(verbose=0)],
            "feature_selection": [RFE(RandomForestClassifier(n_estimators=100, class_weight='balanced'), n_features_to_select=10), RFE(RandomForestClassifier(n_estimators=100, class_weight='balanced'), n_features_to_select=20),
                                  SelectFromModel(RandomForestClassifier(n_estimators=100, class_weight='balanced'), max_features=10), SelectFromModel(RandomForestClassifier(n_estimators=100, class_weight='balanced'), max_features=20)],
          #"estimator__l2_leaf_reg": [3,6],

          "estimator__eta": [0.0001,0.00001, 0.000001],
        "estimator__class_weights": [[1, 9], [1, 15], [1, 3], [1, 6], [1, 30]]
     }]


param_grid_p3b = [
    ## done
    # {
    #     "preprocess": [PowerTransformer()],
    #     'estimator': [SVC()],
    #         "feature_selection": [SelectFromModel(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear',
    #                                                      class_weight='balanced', C=5))],
    #          "feature_selection__max_features": [ 6, 8, 10, 12, 14],
    #         'estimator__tol': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    #     'estimator__C': [1, 5,10,15, 20],
    #     'estimator__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    #     'estimator__class_weight': [{0:1, 1: 6}, {0:1, 1: 2}, {0:1, 1: 3}, {0:1, 1: 4}]},
    # # ## done
    {
          "preprocess": [None],
          "estimator": [CatBoostClassifier(verbose=0)],
            "feature_selection": [RFE(RandomForestClassifier(n_estimators=100, class_weight='balanced'))],
          "feature_selection__n_features_to_select": [22, 14],
          "estimator__l2_leaf_reg": [3, 6],
          "estimator__depth": [5,7,3],
          "estimator__eta": [0.0001,0.00001],
        "estimator__class_weights": [[1,4], [1, 6],[1,2]]
     }]
