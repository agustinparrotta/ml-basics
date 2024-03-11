"""
Problem Set 1: Regressor con scikit-learn
"""
# %%
# -- Imports --
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator

from sklearn.datasets import make_regression
import mlfin.printing as printing
# %%
# -- Define constants --
random_state = 1234

# %%
# -- Class Definition --
class EstimadorOLS(BaseEstimator, RegressorMixin):
    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
            estimation = np.linalg.inv(X.T @ X) @ X.T @ y
            self.coef_ = estimation[1:]
            self.intercept_ = estimation[0]
        else:
            estimation = np.linalg.inv(X.T @ X) @ X.T @ y
            self.coef_ = estimation
            self.intercept_ = 0

        return self

    def predict(self, X):

        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return X@self.coef_ + self.intercept_

# %%
# -- Define Models to use --
lr_ols = EstimadorOLS()

# %%
# -- Check estimator --
# Error: Singular matrix
# check_estimator(EstimadorOLS())

# %%
# -- Generate Training Set --
X, y = make_regression(n_samples=1000, n_features=10, bias=5, noise=10, random_state=random_state)

# %%
# -- Validation --
printing.print_validation_results(lr_ols, X, y, random_state=random_state)

# %%
# -- Fit model to entire training set --
lr_ols.fit(X, y)

print('Fitted coefs:')
print(f' Constant: {lr_ols.intercept_:.6f}')
print(f' {lr_ols.coef_}')
