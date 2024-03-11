"""
Clase 2: Regresiones y NN
"""
# %%
# Imports & definitions
import numpy as np

import sklearn.linear_model as sk_lm
import tensorflow.keras as tfk
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from mlfin.printing import print_validation_results


# %%
# Creando sets de datos
X = np.random.randn(1000, 3)
y = -1. + X @ np.array([3., 5., 7.]) + np.random.randn(1000) * 0.15


# %%
# API Secuencial
nn_seq = tfk.Sequential()
nn_seq.add(tfk.layers.Dense(1, activation='linear', use_bias=True))

nn_seq.compile(optimizer='SGD', loss='mse')
nn_seq.fit(X, y, epochs=5, batch_size=20)
print(nn_seq.get_weights())


# %%
# API Funcional
input_tensor = tfk.Input(shape=(3,))
output_tensor = tfk.layers.Dense(1, use_bias=True)(input_tensor)
nn_func = tfk.Model(input_tensor, output_tensor)

nn_func.compile(optimizer='SGD', loss='mse')
nn_func.fit(X, y, epochs=5, batch_size=20)
print(nn_func.get_weights())


# %%
# Usando KerasWrapper
def build_nn():
    int_layer = tfk.Input(shape=(3,))
    out_layer = tfk.layers.Dense(1, use_bias=True)(int_layer)

    ffn = tfk.Model(int_layer, out_layer)
    ffn.compile(optimizer='SGD', loss='mse')
    return ffn


nn_keras = KerasRegressor(build_fn=build_nn, epochs=5, batch_size=20,
                          verbose=False)


# %%
# Ejecuto Cross-Validation y entreno con API scikit-learn
ols = sk_lm.LinearRegression()

print_validation_results(nn_keras, X, y)
print_validation_results(ols, X, y)

nn_keras.fit(X, y)
ols.fit(X, y)


# %%
# Modelo con 2 capas (1 hidden layer)
input_tensor_2 = tfk.Input(shape=(3,))
hidden = tfk.layers.Dense(2)(input_tensor_2)
output_tensor_2 = tfk.layers.Dense(1)(hidden)

nn_2l = tfk.Model(input_tensor_2, output_tensor_2)
nn_2l.compile(optimizer='SGD', loss='mse')

nn_2l.fit(X, y, epochs=5, batch_size=20)
print(nn_2l.get_weights())


# %%
# Realizamos una predicción
print('Predicción para [1., 4., 6.] = ', nn_2l.predict(np.array([[1., 4., 6.]])))


# %%
# Observamos la red
nn_2l.summary()
