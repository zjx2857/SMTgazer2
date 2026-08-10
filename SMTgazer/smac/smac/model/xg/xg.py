from __future__ import annotations

import os
from typing import Any
import numpy as np
from ConfigSpace import ConfigurationSpace
import lightgbm as lgb
from joblib import Parallel, delayed

from pyrfr import regression
from pyrfr.regression import binary_rss_forest as BinaryForest
from pyrfr.regression import default_data_container as DataContainer
from smac.constants import N_TREES, VERY_SMALL_NUMBER
from smac.model.xg import AbstractXG
from smac.model.random_forest import RandomForest
from smac.model.random_forest import AbstractRandomForest
from warnings import simplefilter


simplefilter(action="ignore",category=FutureWarning)
simplefilter(action="ignore",category=UserWarning)
__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


def lightgbm_device_type() -> str:
    device_type = os.environ.get("SMTGAZER_LIGHTGBM_DEVICE_TYPE", "gpu").lower()
    if device_type not in {"gpu", "cuda"}:
        raise ValueError(
            "SMTGAZER_LIGHTGBM_DEVICE_TYPE must be 'gpu' or 'cuda', "
            f"found {device_type!r}"
        )
    return device_type


class XG(AbstractXG):
    def __init__(
        self,
        configspace: ConfigurationSpace,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.2,
        subsample: float = 0.9,
        colsample_bytree: float =0.9,
        min_child_weight: float = 3,
        reg_alpha: float = 0.8,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        log_y: bool = False,
        min_data_in_leaf: int = 4,
        min_gain_to_split: float = 0.1,
        n_trees: int = N_TREES,
        ratio_features: float = 5.0 / 6.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        forest_depth: int = 2**20,
        bootstrapping: bool = True,
        w1: float = 0.2,
        w2: float = 0.8,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,    
    ) -> None:
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )
        self.model =  lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            reg_alpha = reg_alpha,
            reg_lambda = reg_lambda,
            min_data_in_leaf=min_data_in_leaf,
            min_gain_to_split = min_gain_to_split,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=seed,
            device_type=lightgbm_device_type(),
            verbose=-1
            )
        self.model2=RandomForest(
            log_y=True,
            n_trees=n_trees,
            bootstrapping=bootstrapping,
            ratio_features=ratio_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=forest_depth,
            configspace=configspace,
            instance_features=instance_features,
            seed=seed,
        )
        
        self.weights = [w1,w2]
        self._log_y = log_y
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._min_child_weight = min_child_weight
        self._reg_lambda = reg_lambda
        self._gamma = gamma

    @property
    def meta(self) -> dict[str, Any]:
        return {
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "subsample": self._subsample,
            "colsample_bytree": self._colsample_bytree,
            "min_child_weight": self._min_child_weight,
            "reg_lambda": self._reg_lambda,
            "gamma": self._gamma,
            "pca_components": self._pca_components,
            "seed": self._seed,
        }

    def _train(self, X: np.ndarray, y: np.ndarray) -> XG:
        self.model2._train(X,y)
        X = self._impute_inactive(X)
        self.model.fit(X,y.flatten())
        return self


    def _predict(
            self,
            X: np.ndarray,
            covariance_type: str | None = "diagonal",
        ) -> tuple[np.ndarray, np.ndarray | None]:
        mean2,var2 = self.model2._predict(X,covariance_type)
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != len(self._types):
            raise ValueError("Rows in X should have %d entries but have %d!" % (len(self._types), X.shape[1]))

        if covariance_type != "diagonal":
            raise ValueError("`covariance_type` can only take `diagonal` for this model.")
        X = self._impute_inactive(X)
    
        means = self.model.predict(X)
        stds = []
        for row_X in X:
            stds.append(self._var_threshold)
        means = np.array(means)
        vars_ = np.array(stds)
        means = means.reshape(-1, 1)
        vars_ = vars_.reshape(-1, 1)
        mean_ret = np.stack((means, mean2),axis=0)      
        mean_ret = np.average(mean_ret,axis=0,weights=self.weights) 
        var_ret = var2

        return mean_ret, var_ret

    def predict_marginalized(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean2,var2 = self.model2.predict_marginalized(X)
        if self._n_features == 0:
            mean_, var = self.predict(X)

            var[var < self._var_threshold] = self._var_threshold
            var[np.isnan(var)] = self._var_threshold
            mean_ret = np.stack((mean_, mean2),axis=0)      
            mean_ret = np.average(mean_ret,axis=0,weights=self.weights) 
            var_ret = var2
            
            return mean_ret, var_ret

        assert self._instance_features is not None

        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != len(self._bounds):
            raise ValueError("Rows in X should have %d entries but have %d!" % (len(self._bounds), X.shape[1]))

        
        X = self._impute_inactive(X)

        X_feat = list(self._instance_features.values())
        mean_ = []
        var = []
        merges = []
        for row_X in X:
            
            for feat_ in X_feat:
                merge = np.concatenate((row_X,np.array(feat_)))
                merges.append(merge)
        merges = np.array(merges)
        means = self.model.predict(merges).tolist()
        for i in range(0,len(X)):
            n_feat = len(X_feat)
            mean_.append(np.mean(np.array(means[i*n_feat:(i+1)*n_feat])))
            var.append(self._var_threshold)
        mean_ = np.array(mean_)
        var = np.array(var)

        var[var < self._var_threshold] = self._var_threshold
        if len(mean_.shape) == 1:
            mean_ = mean_.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))
        
        mean_ret = np.stack((mean_, mean2),axis=0)      
        mean_ret = np.average(mean_ret,axis=0,weights=self.weights) 
        var_ret = var2
        
        return mean_ret, var_ret
