"""
Clase 1: Workflow modelo con scikit-learn
"""
# %%
# -- Imports --
import numpy as np
import sklearn.model_selection as sk_ms
import sklearn.linear_model as sk_lm
import sklearn.svm as sk_sv


# %%
# -- Carga de Datos --

# NOTA: En este caso crearemos en forma manual datos sintéticos (veremos luego
#       que sklearn tiene helpers para esto). Típicamente aquí se importan de
#       alguna fuente externa.

np.random.seed(1234)
X = np.random.uniform(size=(1000, 5))
y = 2. + X @ np.array([1., 3., -2., 7., 5.]) + np.random.normal(size=1000)


# %%
# -- Creación de Training y Testing Sets --
X_tr, X_te, y_tr, y_te = sk_ms.train_test_split(X, y, test_size=.1,
                                                random_state=1234)


# %%
# -- Definición de los modelos a usar --
lr = sk_lm.LinearRegression()  # Modelo Lineal
svr = sk_sv.SVR(gamma='scale')  # Support Vector Regression


# %%
# -- Análisis comparativo utilizando técnicas de muestreo (ej. Cross-Validation) --
print('Regresión Lineal')
lr_cv = sk_ms.cross_val_score(lr, X_tr, y_tr, cv=5)
print(lr_cv)
print(f'Mean = {lr_cv.mean():.4f}')
print('\n')
print('Support Vector Regression')
svr_cv = sk_ms.cross_val_score(svr, X_tr, y_tr, cv=5)
print(svr_cv)
print(f'Mean = {svr_cv.mean():.4f}')


# %%
# -- Evaluación FINAL del modelo ganador sobre el Testing Set --

# NOTA: Este paso NO implica volver a estimar todo si el poder predictivo es
#       malo, sino simplemente entender cómo se comporta nuestro mejor modelo
#       out-of-sample. Si la performance no es la deseada lamentablemente no
#       debería ser aplicado. Habría que seguir investigando y esperar la
#       existencia de nuevos datos para volver a aplciar el workflow.

lr.fit(X_tr, y_tr)
print(f'Testing Set Score : {lr.score(X_te, y_te):.2%}')


# %%
# Entreno sobre dataset completo para poner en producción

# NOTA: Sólo se realiza si la performance final cumple con el umbral deseado.

lr.fit(X, y)

print(f'Coeficientes = {lr.coef_}')
print(f'Intercepto = {lr.intercept_}')
print('Prediction para [1., 4., 6., 0., -2.] = ', end='')
print(lr.predict([[1., 4., 6., 0., -2.]]))
