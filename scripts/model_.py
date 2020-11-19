import os
import sys
import glob
import shutil
import multiprocessing

import numpy as np 
import pandas as pd 
from scipy.io import loadmat

import multiprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# import lightgbm as lgb
from sklearn import naive_bayes
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA 
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from imblearn.combine import SMOTETomek
from sklearn.manifold import TSNE

from sklearn.decomposition import LatentDirichletAllocation


import umap 

# from cuml.manifold.umap import UMAP as cumlUMAP
from sklearn.manifold.t_sne import trustworthiness


# import tensorflow as tf 

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import TimeDistributed
# from tensorflow.keras.layers import Conv1D
# from tensorflow.keras.layers import MaxPooling1D


ncores = multiprocessing.cpu_count() - 1


# def smote_to_balance_df(X_train, y_train):
#     # sm = SMOTE(random_state=42)
#     sm = SMOTE(random_state=42, k_neighbors=5)

#     X_train2, y_train_2 = sm.fit_resample(X_train, y_train)

#     return X_train2, y_train_2


def agg_clustering(X):

	return AgglomerativeClustering(n_clusters=2).fit_predict(X)


def kmeans_clustering(X):

	return KMeans(n_clusters=4).fit(X)


def rf_classifier(X, y):

    return RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=ncores).fit(X, y)


def xgboost_classifier(X, y):
    xgboost = XGBClassifier(learning_rate=0.1, n_estimators=100, n_jobs=ncores) # , max_depth=32 # for pain detection usnig scale_pos_weight = 2

    model = xgboost.fit(X, y)

    return model 


def knn_classifier(X, y):

    return KNeighborsClassifier(n_neighbors=5, n_jobs=ncores).fit(X, y)


def svm_classifier(X, y):

    return SVC(C=1.0, kernel='rbf', gamma='scale', class_weight="balanced", probability=True).fit(X, y)


def naive_Bayes_classifier(X, y):

	return naive_bayes.GaussianNB().fit(X, y)


def elastic_net_classifier(X, y):

	return ElasticNet().fit(X, y)


def logistic_regr_classifier(X, y):

	return LogisticRegression().fit(X, y)


def pca(X, n_comp):

	return PCA(n_components=n_comp).fit(X).transform(X)


def latent_da(X, n_comps):
    lda = LatentDirichletAllocation(n_components=n_comps, random_state=2020) # n_jobs=None

    return lda.fit_transform(X)


def latent_da_v2(X_train, X_test, n_comps):
    lda = LatentDirichletAllocation(n_components=n_comps, random_state=2020) # n_jobs=None 

    return lda.fit_transform(X_train), lda.transform(X_test)




# def t_SNE(X):
#     embedd = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, 
#                     n_iter=1000, metric='euclidean', random_state=2020, method='barnes_hut', n_jobs=ncores)

#     return embedd.fit_transform(X)


def umap_(X, n_neigh, m_dist):
    reducer = umap.UMAP(n_neighbors=n_neigh, min_dist=m_dist, n_components=2)

    return reducer.fit_transform(X)


# def cuml_UMAP_(X, n_neigh, m_dist):
#     reducer = cumlUMAP(n_neighbors=n_neigh, min_dist=m_dist, n_components=2)

#     return reducer.fit_transform(X)


def rf_regr(X, y):

    return RandomForestRegressor(n_estimators=100, n_jobs=ncores).fit(X, y)


# def CNN_LSTM_classifier(X_train, y_train, X_test, y_test):
#     verbose, epochs, batch_size = 0, 20, 32
#     samp_leng, mod_feature, n_classes = X_train.shape[1], X_train.shape[2], y_train.shape[1]
#     steps, leng = 5, 35 # samp_leng // 35, samp_leng // 5 

#     X_train = X_train.reshape(X_train.shape[0], steps, leng, mod_feature)
#     X_test = X_test.reshape(X_test.shape[0], steps, leng, mod_feature)

#     # define model 
#     model = Sequential()

#     model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu'), input_shape=(None, leng, mod_feature)))
#     model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu')))
#     model.add(TimeDistributed(Dropout(0.2)))  
#     model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

#     model.add(TimeDistributed(Flatten()))
    
#     model.add(LSTM(16))
#     model.add(Dropout(0.2))
    
#     model.add(Dense(64, activation='relu'))
    
#     model.add(Dense(n_classes, activation='softmax')) # classification layers 

#     print(model.summary())

#     opt = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6)

#     model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

#     print(model.history)

#     pred = model.predict(X_test).argmax(axis=-1)

#     _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
#     print(accuracy) 

#     return model, pred 



# def evaluate_model(trainX, trainy, testX, testy):
#     # define model
#     verbose, epochs, batch_size = 0, 25, 64
#     n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
#     # reshape data into time steps of sub-sequences
#     n_steps, n_length = 4, 32
#     trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
#     testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
#     # define model
#     model = Sequential()
#     model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
#     model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
#     model.add(TimeDistributed(Dropout(0.5)))
#     model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
#     model.add(TimeDistributed(Flatten()))
#     model.add(LSTM(100))
#     model.add(Dropout(0.5))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(n_outputs, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # fit network
#     model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
#     # evaluate model
#     _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
#     return accuracy


if __name__ == "__main__":
	pass 
