# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:36:51 2022

@author: ariel
"""

# =============================================================================
# Regresion SVR
# =============================================================================

#Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values 

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))


#Importar la libreria para SVR
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X, y)

#Prediccion de nuestro modelo con SVR
y_pred = regression.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)

#Revertir escalado
X_inv = sc_X.inverse_transform(X)
y_inv = sc_y.inverse_transform(y.reshape(-1,1))

#Grafico
plt.scatter(X_inv, y_inv, color = 'red')
plt.plot(X_inv,sc_y.inverse_transform(regression.predict(X)), color = 'blue')
plt.title('Modelo de SVR')
plt.xlabel('Posicion del empleado')
plt.ylabel('Sueldo (en dolares)')
plt.show()