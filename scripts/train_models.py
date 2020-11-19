import os
import sys
import multiprocessing
from itertools import islice
from random import randint

import pandas as pd
import numpy as np
from scipy import stats
from scipy import signal 
from statsmodels import robust
import pickle

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

# import tensorflow as tf 
# from tensorflow.keras.utils import to_categorical

from metrics_ import eval_regression, eval_classification, ccc, pcc, accuracy, precision, recall, f1score, auc_roc, cohen_kappa, mcc # import custom evaluation metrics 
from sklearn.metrics import confusion_matrix, classification_report

import joblib 

# custom import 
import model_
import metrics_


import os
import sys
import multiprocessing
from itertools import islice
from random import randint

import pandas as pd
import numpy as np
from scipy import stats
from scipy import signal 
from statsmodels import robust
import pickle

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

# import tensorflow as tf 
# from tensorflow.keras.utils import to_categorical

from metrics_ import eval_regression, eval_classification, ccc, pcc, accuracy, precision, recall, f1score, auc_roc, cohen_kappa, mcc # import custom evaluation metrics 
from sklearn.metrics import confusion_matrix, classification_report

import joblib 

# custom import 
import model_
import metrics_



def train_n_cv_valid_models(df, features, flag):
    folds = df.folds.unique().tolist()
    print(folds)

    res_lst = []
    fold_count = 0
    m_res_mat = np.zeros((0, 10)) # 8 is subject to change 
    for fold in folds:
        train_df = df[df["folds"].isin([fold]) == False]
        test_df = df[df["folds"].isin([fold]) == True]
        
        print(train_df.shape)
        
        # run experiment for classification with BERT features 
        if flag == "XGB":
            model = model_.xgboost_classifier(train_df[features].astype(np.float64), train_df["ground_truth"])
            pred = model.predict(test_df[features])

            clf_report = classification_report(test_df["ground_truth"], pred, output_dict=True)
            cls_report_df = cls_report_dict2mat(clf_report)
            print("Results obtained on fold:", fold)
            print(cls_report_df)

        # TODO: include tweet ID and prediction for analysis 

    return 0 


def train_n_cv_valid_LDA(df, features, flag):
    folds = df.folds.unique().tolist()
    print(folds)

    res_lst = []
    fold_count = 0
    n_comps = 10
    m_res_mat = np.zeros((0, 11)) # 8 is subject to change 
    for fold in folds:
        train_df = df[df["folds"].isin([fold]) == False]
        test_df = df[df["folds"].isin([fold]) == True]
        
        print(train_df.shape)
        
        if flag == "LDA":
            
            _, X_test = model_.latent_da_v2(train_df[features].astype(np.float64), test_df[features].astype(np.float64), n_comps)
            print("X_test", X_test.shape)

            mat = np.concatenate([X_test, test_df[["ID"]].values], axis=1)
            print(mat.shape)


            m_res_mat = np.append(m_res_mat, mat, axis=0)
            print("LDA matrix:", m_res_mat.shape)

    return pd.DataFrame(m_res_mat, columns=["LDA_comp_" + str(i+1) for i in range(n_comps)] + ["ID"])



def cls_report_dict2mat(cls_report_dict):
    """
    columns -> precision, recall, f1-score, and support
    row class, accuracy, ... 
    """

    return pd.DataFrame(cls_report_dict)



# def train_n_rand_validate_model(df, aid_flag, features, model_type, wrt_dir, model_name):
#     # count_ = 0
#     n_seed = 5
#     m_res_mat = np.zeros((0, 10)) # 8 is subject to change 
#     for seed_ in range(n_seed): # validation 4 times 
#         # train_df = df[df["subject"].isin(fold) == False]
#         # test_df = df[df["subject"].isin(fold) == True]

#         # count_ += 1 
#         train_df, test_df, y_train_vec, y_test_vec = train_test_split(df, df["ground_truth"], 
#                                                                     test_size=0.3, # 0.3, 0.95
#                                                                     random_state=seed_, 
#                                                                     stratify=df["month"])


#         X_train, y_train = train_df[features], train_df["groundTruth"]
#         X_test, y_test = test_df[features], test_df["groundTruth"]

#         print("Number of features used in model training:", X_train.shape)

#         # train model 
#         if model_type == "rf":
#             X_train = X_train.astype(float)
#             X_test = X_test.astype(float)
#             X_train, X_test = min_max_norm(X_train, X_test, "")


#            clf_report = classification_report(orig, pred, output_dict=True)
#            print(clf_report)

#     return 0 



if __name__=="__main__":
    data_dir = "../data/stack_files"
    # wrt_dir = "../data/stack_files"

    proMask_filename = "AntiMask_raw_data.csv"
    antiMask_filename = "ProMask_raw_data.csv"
    antiMask_df = pd.read_csv(os.path.join(data_dir, antiMask_filename))
    proMask_df = pd.read_csv(os.path.join(data_dir, proMask_filename))

    m_df = pd.concat([proMask_df, antiMask_df])
 