from imblearn.over_sampling import BorderlineSMOTE, SMOTE

from sklearn.naive_bayes import GaussianNB


from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler





param_grid = [
    ## done
    {
        "preprocess": [PowerTransformer()],
        "feature_selection": [RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear',
                                                      C=15)),
                              RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear',
                                                     class_weight='balanced')),
                              RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear', C=5))],
        "feature_selection__n_features_to_select": [6, 8,  10],
        "sampling": [BorderlineSMOTE(),None],
        'estimator': [SVC()],
        'estimator__C': [1, 5,10,15],
        'estimator__kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
        'estimator__class_weight': [None, 'balanced', {0:1, 1:2}]},
    # ## done
    {
          "preprocess": [PowerTransformer()],
          "feature_selection": [RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear', C=5, class_weight='balanced'))],
          "feature_selection__n_features_to_select": [6],
          "sampling": [None],
          "estimator": [CatBoostClassifier(verbose=0)],
          # "estimator__l2_leaf_reg": [0,  1, 10],
          #"estimator__depth": [4,6,8],
          #"estimator__eta": [0.01, 0.0001],
        #  "estimator__class_weights": [[1, 5]]
     },
    # ## done
     {
       "preprocess": [PowerTransformer()],
         "feature_selection": [RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear',
                                                      class_weight='balanced', C=5))],
         "feature_selection__n_features_to_select": [4, 8],
         "sampling": [BorderlineSMOTE(), None],
         "estimator": [XGBClassifier()],
         #'estimator__n_estimators': [150, 500],
         'estimator__learning_rate': [0.01, 0.1, 0.5],
         # 'estimator__subsample': [0.3, 0.7],
         #'estimator__scale_pos_weight': [1, 6],
         #'estimator__min_child_weight': [2, 4]
     },
    # ### done
    # {
    #     "preprocess": [PowerTransformer()],
    #     "feature_selection": [RFE(LogisticRegression(max_iter=5000, penalty="l1", solver='liblinear',
    #                                                  class_weight='balanced', C=5))],
    #     "feature_selection__n_features_to_select": [4, 8, 12],
    #     "sampling": [BorderlineSMOTE(), None],
    #     "estimator": [GaussianNB()],
    # },
    #
    # ## done
    {
         "preprocess": [PowerTransformer()],
         "feature_selection": [RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear',
                                                      class_weight='balanced', C=5))],
         "feature_selection__n_features_to_select": [6, 8],
        "sampling": [BorderlineSMOTE(), None],
         "estimator": [RandomForestClassifier()],
         "estimator__n_estimators": [100, 500],
         "estimator__max_depth": [3, 8],

     }]

