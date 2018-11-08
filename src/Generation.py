# coding: utf-8
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class XgboostFeature():
    ##可以传入xgboost的参数
    ##常用传入特征的个数 即树的个数 默认30
    def __init__(self, n_estimators=30, learning_rate=0.3, max_depth=3, min_child_weight=1, gamma=0.3, subsample=0.8,
                 colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1, reg_alpha=1e-05,
                 reg_lambda=1, seed=27):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.nthread = nthread
        self.scale_pos_weight = scale_pos_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.seed = seed
        print
        'Xgboost Feature start, new_feature number:', n_estimators

    def mergeToOne(self, X, X2):
        X3 = []
        for i in range(X.shape[0]):
            tmp = np.array([list(X.iloc[i]), list(X2[i])])
            X3.append(list(np.hstack(tmp)))
        X3 = np.array(X3)
        return X3

    ##整体训练
    def fit_model(self, data, target, test):
        clf = XGBRegressor(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            nthread=self.nthread,
            scale_pos_weight=self.scale_pos_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            seed=self.seed)
        data = np.array(data).astype(float)
        scaler = MinMaxScaler()
        temp = scaler.fit(data)
        data = scaler.transform(data)
        test = scaler.transform(test)
        target = scaler.fit_transform(target)

        clf.fit(data, target)
        new_feature = clf.apply(data)
        new_test = clf.apply(test)
        X_train_new = self.mergeToOne(pd.DataFrame(data), new_feature)
        X_test_new = self.mergeToOne(pd.DataFrame(test), new_test)
        X_train_new = pd.DataFrame(X_train_new)
        X_test_new = pd.DataFrame(X_test_new)
        return X_train_new, target, X_test_new
