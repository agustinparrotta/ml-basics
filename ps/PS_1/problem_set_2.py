"""
Problem Set 1: Regressor con scikit-learn
"""
# %%
# -- Imports --
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator

from sklearn.datasets import make_regression
import mlfin.printing as printing

# %%
# -- Define constants --
random_state = 1234
# %%
# -- Class Definition --
class EstimadorOLS(LinearRegression):
    def __init__(self, *, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)

    def fit(self, X, y):
        return super().fit(X, y)

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        return super().score(X, y)

    def get_params(self, deep=True):
        return super().get_params(deep=deep)

    def set_params(self, **params):
        return super().set_params(**params)

# %%
# -- Class Definition --
# class EstimadorOLS2(BaseEstimator, RegressorMixin):

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            self.const_ = 1
        else:
            self.const_ = y.mean()

        d = X.shape[1]
        mse_loss = lambda coefs: np.mean(np.power(y - X@coefs[:-1] - coefs[-1], 2))
        *self.coef_, self.intercept_ = minimize(mse_loss, x0=np.array((d+1)*[0.])).x

        # Return the classifier
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
check_estimator(EstimadorOLS())

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
