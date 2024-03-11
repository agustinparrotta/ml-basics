# %%
# -- Imports --
import pandas as pd
from sklearn.linear_model import LassoCV
from datetime import date, datetime
from mlfin.utils import get_allocations
from sklearn.metrics import mean_squared_error
import numpy as np

# %%
# -- Preprocess DF --
def preprocess(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
    df.set_index(date_col, inplace=True)
    df.dropna(inplace=True)
    return df

# %%
# -- Define Class --
class Portfolio():
    def __init__(self, d_etfs):
        self.d_etfs = d_etfs

    def get_balanced_portfolio(self, d_factors):

        num_factors = self.d_etfs.shape[1]
        baseline = np.full((num_factors,), 1/num_factors)

        optimum = 10000
        optimum_params = {}

        for weights in get_allocations(num_factors):

            y = self.d_etfs @ weights
            X = d_factors

            model_lassocv = LassoCV(cv=5).fit(X, y)
            coefs = model_lassocv.coef_

            mse = mean_squared_error(baseline, coefs)

            if mse < optimum:
                optimum = mse
                optimum_params['coefs'] = coefs
                optimum_params['weights'] = weights

        return (optimum_params['weights'], optimum_params['coefs'])

# %%
# -- Testing --
if __name__ == '__main__':

    selected_etfs = pd.read_csv('ps/PS_2/data/selected_etfs.csv')
    data_factors = pd.read_csv('ps/PS_2/data/F-F_Research_Data_5_Factors_2x3_daily.csv')

    assets = ['DATE', 'BOND', 'SUSA', 'DNL', 'XLF', 'XSLV']
    factors = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']


    selected_etfs = preprocess(selected_etfs[assets], 'DATE')
    data_factors = preprocess(data_factors[factors], 'Date')

    data_etfs = ((selected_etfs / selected_etfs.shift(1) -1) * 100)[1:]

    d_etfs = data_etfs.loc['2017-01-01':'2017-12-31']
    d_factors = data_factors.loc['2017-01-01':'2017-12-31']
    my_portfolio = Portfolio(d_etfs)

    optimum_weights, optimum_coefs = my_portfolio.get_balanced_portfolio(d_factors)

    print('--- Portfolio Balanceado ---')
    print(f'Weights   : {["{:0.2%}".format(x) for x in optimum_weights]}')
    print(f'Exposures : {["{:0.2%}".format(x) for x in optimum_coefs]}')
