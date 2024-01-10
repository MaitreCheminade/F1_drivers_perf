import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_percentage_error, explained_variance_score
from sklearn.inspection import PartialDependenceDisplay, partial_dependence


# params
df_path = "../../tmp/working_data/filtered.csv"
binary_path = "../../tmp/working_data/binaries.csv"
df = pd.read_csv(df_path, sep=";", header=0, encoding="UTF-8")
binary_df = pd.read_csv(binary_path, sep=";", header=0, encoding="UTF-8").drop("Unnamed: 0", axis=1)
# If you want to only use one track for either of the dataframes
# df = df.loc[df["EventName"] == "Brazilian Grand Prix"]
# binary_df = binary_df.loc[binary_df["EventName_Brazilian Grand Prix"] == 1]


# process the variables
def process_variables_for_ridge():
x = binary_df.iloc[:, :]
x = x.drop("LapTime", axis=1)
y = binary_df["LapTime"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=24)
variable_names = x.columns

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# créer le modèle
alpha = 0.1  # Paramètre de régularisation
degree = 2
ridge_model = make_pipeline(PolynomialFeatures(degree, interaction_only=True), Ridge(alpha=alpha, solver="saga"))

# entraîner le modèle aux données
ridge_model.fit(x_train, y_train)

# évaluer la qualité du modèle sur le subset d'entrainement
y_train_predict = ridge_model.predict(x_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
absolute_train = mean_absolute_percentage_error(y_train, y_train_predict)
r2 = r2_score(y_train, y_train_predict)

print("La performance du Modèle pour le set de Training")
print("------------------------------------------------")
print('le score R2 est {:.3}'.format(r2))
print("l'erreur RMSE est {:.3}".format(rmse))
print("le pourcentage absolu d'erreur moyenne est {:.3}".format(absolute_train))
print("\n")

# évaluation du modèle pour le set de test
y_test_predict = ridge_model.predict(x_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
absolute_test = mean_absolute_percentage_error(y_test, y_test_predict)
r2_test = r2_score(y_test, y_test_predict)

print("La performance du Modèle pour le set de Test")
print("--------------------------------------------")
print('le score R2 est {:.3}'.format(r2_test))
print("l'erreur RMSE est {:.3}".format(rmse_test))
print("le pourcentage absolu d'erreur moyenne est {:.3}".format(absolute_test))
print("\n")

# Get coefficients and corresponding variable names
variable_names = x.columns
# Get the final estimator from the pipeline
final_estimator = ridge_model.steps[-1][1]
# Get coefficients from the final estimator
var_coef = final_estimator.coef_
# Print the coefficients
print("These are the effects of your variables on lap times :")
print("------------------------------------------------------")
for variable, coef in zip(variable_names, var_coef):
    print(f"{variable}: {coef:.4f}")



