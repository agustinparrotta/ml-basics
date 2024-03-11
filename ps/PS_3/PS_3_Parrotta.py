# %%
# -- Imports --
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings(action='ignore')

# %%
# -- Define Class --
class AnalistaDeRiesgo():
    def __init__(self, models_conf: list[tuple[None]], score: str) -> None:
        self.models_conf = models_conf
        self.score = score
        return None
    
    def load_data(self, df_train: pd.DataFrame, label: str) -> None:
        self.df_train = df_train
        self.label = label
        return None

    def get_report(self, df_predict: pd.DataFrame, features: list[str]) -> None:
        best_models = []

        for model_conf in self.models_conf:
            model = model_conf[0]
            params = model_conf[1]

            gs = GridSearchCV(model, params, cv=5, scoring=self.score)
            gs.fit(self.df_train[features], self.df_train[self.label])
        
            best_models.append((gs.best_score_, gs.best_estimator_))

        best_model = max(best_models,key=lambda x: x[0])[1]

        predictions = best_model.predict(df_predict[features])

        num_defaults = predictions.sum()
        num_assets = predictions.size
        per_defaults = num_defaults / num_assets

        print(f"Entidades en riesgo de default = {num_defaults:.0f}")
        print(f"Total de activos del sistema (USD B) = {num_assets:.0f}")
        print(f"Porcentaje de activos en riesgo de default = {per_defaults:.2%}")

        return predictions

# %%
# -- Testing --
if __name__ == '__main__':

    # -- Import Data --
    df_bank_defaults = pd.read_hdf('data/central_bank_data.h5', key='bank_defaults_FDIC')
    df_regulated_banks = pd.read_hdf('data/central_bank_data.h5', key='regulated_banks')

    # -- Define Columns --
    columns = ['log_TA', 'NI_to_TA', 'Equity_to_TA', 
                        'NPL_to_TL', 'REO_to_TA', 'ALLL_to_TL', 
                        'core_deposits_to_TA', 'brokered_deposits_to_TA', 'liquid_assets_to_TA', 
                        'loss_provision_to_TL','NIM', 'assets_growth', 
                        'defaulter']

    feature_columns = ['log_TA', 'NI_to_TA', 'Equity_to_TA', 
                        'NPL_to_TL', 'REO_to_TA', 'ALLL_to_TL', 
                        'core_deposits_to_TA', 'brokered_deposits_to_TA', 'liquid_assets_to_TA', 
                        'loss_provision_to_TL','NIM', 'assets_growth']
    
    label_column = 'defaulter'

    # -- Select and Process Data --
    df_bank_defaults = df_bank_defaults[columns]
    df_regulated_banks = df_regulated_banks[columns]

    df_bank_defaults.dropna(inplace=True)
    df_regulated_banks.dropna(inplace=True)

    # -- Models --
    model_knn = KNeighborsClassifier()

    param_knn = {
        'n_neighbors': range(3, 31)
    }

    model_svc = SVC()

    param_svc = {
        'C': [1, 10, 100, 500, 1000],
        'gamma': ['scale'],
        'kernel': ['linear', 'rbf']
    }

    model_tree = DecisionTreeClassifier()

    param_tree = {
        'min_samples_split': range(2, 16)
    }

    # -- Create Instance --
    risk = AnalistaDeRiesgo(
        [(model_knn, param_knn), (model_svc, param_svc), (model_tree, param_tree)],
        'roc_auc'
    )

    # -- Call load_data --
    risk.load_data(df_bank_defaults, label_column)

    # -- Call get_report --
    predictions = risk.get_report(df_regulated_banks, feature_columns)
