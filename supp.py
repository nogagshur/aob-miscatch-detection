from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, roc_curve, balanced_accuracy_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import os

import pandas as pd
import numpy as np


# Mrmr , fcbf, SBS, extra trees


def calc_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity


def create_scores_row(model_name, sample_name, y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensetivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_true, y_pred)
    # bal_acc = balanced_accuracy_score(y_true, y_pred)

    d = {
        'estimator': model_name,
        'f1_score': f1,
        'sensetivity': sensetivity,
        'specificity': specificity,
        'roc_auc_score': auc,
        'sample': sample_name
    }
    # keys = list(d.keys())
    # for key in keys:
    #     if key is 'estimator':
    #         continue
    #     d[sample_name + '_' + key] = d.pop(key)
    return pd.to_numeric(pd.Series({**d}), errors='ignore')


def summarize_cv(self):
    return pd.Series({})


def create_roc_fig(X, y, pipeline, out_path, name, save_fig=True):
    y_proba = pipeline.predict_proba(X)
    y_proba = y_proba[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    auc_score = auc(fpr, tpr)

    # plot no skill
    if save_fig:
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.')
        for i, j, txt in zip(fpr[1::3], tpr[1::3], thresholds[1::3]):
            plt.annotate(np.round(txt, 2), (i, j - 0.04), fontsize=7)
        # show the plot
        # plt.show()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve (AUC = {auc_score:.3f})')
        if not os.path.exists(os.path.join(out_path, 'roc_figs')):
            os.makedirs(os.path.join(out_path, 'roc_figs'))
        plt.savefig(os.path.join(out_path, 'roc_figs', name + '_roc.png'))
        plt.close()

    return fpr, tpr, auc_score


def create_roc_cv_fig(tprs, aucs, out_path, name):
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve CV')
    plt.legend(loc="lower right")
    # plt.show()

    if not os.path.exists(os.path.join(out_path, 'roc_figs')):
        os.makedirs(os.path.join(out_path, 'roc_figs'))
    plt.savefig(os.path.join(out_path, 'roc_figs', name + '_roc.png'))
    plt.close()


def decision_maker(row):
    # receives a row from DF usage-> df.apply(decision_maker,axis=1)
    # first column must be frequent
    if (np.sum(row.values == 1) == 1) & (row.values[0] <= 0):
        return 'review'
    elif 1 in row.values:
        return 'nonpass'
    elif 0 in row.values:
        return 'review'
    else:
        return 'pass'


# def combine_models(self,models,X):
def find_thresh(fpr, fnr, thresholds, error):
    # finds the thresholds that give the 'error' for each side
    tmp_thresh = thresholds[fpr <= error]
    fpr_ind = np.argmin(tmp_thresh)
    upper_thresh = tmp_thresh[fpr_ind]
    tmp_thresh = thresholds[fnr <= error]
    fnr_ind = np.argmax(tmp_thresh)
    lower_thresh = tmp_thresh[fnr_ind]
    return lower_thresh, upper_thresh


def create_fn_fp_fig(y, y_proba, out_path, name, save_fig=True):
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    thresholds = thresholds[1:]
    fprn = []
    tprn = []
    fnrn = []
    for i, t in enumerate(thresholds):
        new_predict = np.where(y_proba > t, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y, new_predict).ravel()
        fprn.append(fp / (tn + fp))
        tprn.append(tp / (tp + fn))
        fnrn.append(1 - tprn[i])
    fprn = np.array(fprn)
    fnrn = np.array(fnrn)
    # find the threshold that minimizes FPR and FNR
    ind = np.argmin(abs(fprn - fnrn))
    opt_thresh = thresholds[ind]

    one_p_fnr, one_p_fpr = find_thresh(fprn, fnrn, thresholds, error=0.01)
    five_p_fnr, five_p_fpr = find_thresh(fprn, fnrn, thresholds, error=0.05)
    plt.plot(thresholds, fprn, label='FPR')
    plt.plot(thresholds, fnrn, label='FNR')
    plt.annotate(np.round(opt_thresh, 2), (opt_thresh, fprn[ind] - 0.02), fontsize=7)
    plt.xlabel('Thresholds')
    plt.title(f'FPR and FNR vs thresholds \n 1% thresholds:[{one_p_fnr:.2f}, {one_p_fpr:.2f}] \n '
              f'5% thresholds:[{five_p_fnr:.2f}, {five_p_fpr:.2f}]')
    plt.legend(loc="lower right")

    if not os.path.exists(os.path.join(out_path, 'FPR_FNR')):
        os.makedirs(os.path.join(out_path, 'FPR_FNR'))
    plt.savefig(os.path.join(out_path, 'FPR_FNR', name + '_FPR_FNR.png'))
    plt.close()
    return {'one_p_thresh': (one_p_fnr, one_p_fpr),
            'five_p_thresh': (five_p_fnr, five_p_fpr)
            }


def compute_thresh(y, y_proba, allowed_error, type):
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    thresholds = thresholds[1:]
    fprn = []
    tprn = []
    fnrn = []
    for i, t in enumerate(thresholds):
        new_predict = np.where(y_proba > t, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y, new_predict).ravel()
        fprn.append(fp / (tn + fp))
        tprn.append(tp / (tp + fn))
        fnrn.append(1 - tprn[i])
    fprn = np.array(fprn)
    fnrn = np.array(fnrn)
    th = find_thresh(fprn, fnrn, thresholds, error=allowed_error)
    if type is 'FP':
        th = th[1]
    elif type is 'FN':
        th = th[0]
    else:
        raise ('threshold type is not supported')
    return th


def print_conf_mat(true_label, pred):
    df_confusion = pd.crosstab(true_label, pred)
    df_confusion_norm = pd.crosstab(true_label, pred, normalize=True).round(
        4) * 100

    print('\nconfusion matrix:')
    print(df_confusion)
    print('\nconfusion matrix normalized (%):')
    print(df_confusion_norm)


'''
 def apply_fs_clf(self):
        for key, value in self.fs_dict['estimators'].items():
            print(key)
            value.set_params(**self.fs_dict['params'][key])
            pipeline = Pipeline([
                ('feature_selection', SelectFromModel(value)),
                ('classification', model)
            ])
            pipeline.fit(X_train_oversampled, y_train_oversampled)
            y_pred_fs = pipeline.predict(X_test_cv)
            row = fs_obj.create_scores_row(name + '_' + key, 'test', y_test_cv, y_pred_fs)
            rows.append(row)

    # def get_features(self,X,Y,strategy='LinearSVC'):
    #
    #     est = fs_dict['estimators'][strategy]
    #     est.set_params(**fs_dict['params'][strategy])
    #     clf = est.fit(X_train_oversampled,y_train_oversampled)
    #     model = SelectFromModel(est, prefit=True)
    #     selected_f =
    #     X_new = model.transform(X_train_oversampled)
    #     X_new.shape
    #     return

 '''