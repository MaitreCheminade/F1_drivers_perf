from sklearn.neural_network import _multilayer_perceptron, MLPRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import ridge_regression, LinearRegression, LogisticRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import sys

# params
df_path = "../tmp/working_data/filtered.csv"
binary_path = "../tmp/working_data/binaries.csv"
df = pd.read_csv(df_path, sep=";", header=0, encoding="UTF-8")
binary_df = pd.read_csv(binary_path, sep=";", header=0, encoding="UTF-8").drop("Unnamed: 0", axis=1)
# If you want to only use one track for either of the dataframes
# df = df.loc[df["EventName"] == "Brazilian Grand Prix"]
# binary_df = binary_df.loc[binary_df["EventName_Brazilian Grand Prix"] == 1]


# process the variables
x = binary_df.iloc[:, :]
x = x.drop("LapTime", axis=1)
y = binary_df["LapTime"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=24)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# créer le modèle
alpha = 0.05
reg = MLPRegressor(random_state=1, activation="relu", max_iter=150, alpha=alpha, learning_rate="constant", solver="adam").fit(x_train, y_train)

# tester les hyperparamètres
param_grid = {
    'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
    'max_iter': [50, 100],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# grid = GridSearchCV(reg, param_grid, n_jobs=-1, cv=5)
# grid.fit(x_train, y_train)
# sys.exit()
# print("These are the best params :", grid.best_params_)
"""Result : These are the best params : 
{'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (150, 100, 50), 'learning_rate': 'constant', 'max_iter': 50, 'solver': 'adam'}"""

# évaluer la qualité du modèle sur le subset d'entrainement
y_train_predict = reg.predict(x_train)
# racine carrée de l'erreur quadratique moyenne du modèle
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)
# on imprime les résultats de cette évaluation
print("La performance du Modèle pour le set de Training")
print("------------------------------------------------")
print("l'erreur RMSE est {:.3}".format(rmse))
print('le score R2 est {:.3}'.format(r2))
print("\n")

# évaluation du modèle pour le set de test
y_test_predict = reg.predict(x_test)
# racine carrée de l'erreur quadratique moyenne du modèle
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
# score R carré du modèle
r2_test = r2_score(y_test, y_test_predict)
print("La performance du Modèle pour le set de Test")
print("--------------------------------------------")
print("l'erreur RMSE est {:.3}".format(rmse_test))
print('le score R2 est {:.3}'.format(r2_test))
print("\n")

# Get coefficients and corresponding variable names
variable_names = x.columns

# Get coefficients and corresponding variable names
linear_coefficients = reg.coefs_

# Print the coefficients
print("These are the effects of your variables on lap times :")
print("------------------------------------------------------")
for variable, coef in zip(variable_names, linear_coefficients):
    print(f"{variable}: {coef:}")
