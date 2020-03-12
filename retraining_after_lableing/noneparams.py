
from sklearn.naive_bayes import GaussianNB


from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler


param_grid_non_smote = [
    ## done
    {
        "preprocess": [StandardScaler(), PowerTransformer(), None],
        "feature_selection": [RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear',
                                                     class_weight='balanced')),
                              RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear', C=5))],
        "feature_selection__n_features_to_select": [3, 5, 7],
        'estimator': [SVC()],
        'estimator__C': [1, 10],
        'estimator__kernel': ('linear', 'rbf'),
        'estimator__class_weight': [None, 'balanced']},
    ## done
    {
        "preprocess": [PowerTransformer(), None],
        "feature_selection": [RFE(LogisticRegression(penalty="l1", solver='liblinear', class_weight='balanced')),
                              RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear', C=5))],
        "feature_selection__n_features_to_select": [3, 5, 7],
        "estimator": [CatBoostClassifier(verbose=0)],
        # "estimator__l2_leaf_reg": [0,  1, 10],
        "estimator__depth": [4,6,8],
        "estimator__eta": [0.005, 0.01, 0.0001],
        "estimator__class_weights": [[1, 1], [1, 5], [1, 10]]
    },
    ## done
    {
        "preprocess": [PowerTransformer(), None],
        "feature_selection": [RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear',
                                                     class_weight='balanced')),
                              RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear', C=5))],
        "feature_selection__n_features_to_select": [3, 5, 7],
        "estimator": [XGBClassifier()],
        'estimator__n_estimators': [150, 500],
        'estimator__learning_rate': [0.01, 0.1, 0.5],
        'estimator__subsample': [0.3, 0.7],
        'estimator__scale_pos_weight': [1, 3, 7],
        'estimator__min_child_weight': [2, 4]
    },
    ### done
    {
        "preprocess": [StandardScaler(), PowerTransformer(), None],
        "feature_selection": [ RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear',
                                                     class_weight='balanced')),
                              RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear', C=5))],
        "feature_selection__n_features_to_select": [3, 5, 7],
        "estimator": [GaussianNB()],
    },

    ## done
    {
        "preprocess": [StandardScaler(), PowerTransformer(), None],
        "feature_selection": [ RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear',
                                                     class_weight='balanced')),
                              RFE(LogisticRegression(max_iter=500, penalty="l1", solver='liblinear', C=5))],
        "feature_selection__n_features_to_select": [3, 5, 7],
        "estimator": [RandomForestClassifier()],
        "estimator__min_samples_split": [2, 4],
        "estimator__n_estimators": [100, 500],
        # "estimator__max_depth": [3, 7, 10, 20],

    }

]
