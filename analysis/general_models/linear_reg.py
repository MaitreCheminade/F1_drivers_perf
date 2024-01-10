import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    ridge_regression,
    LinearRegression,
    LogisticRegression,
    Lasso,
)
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.svm import SVR

# TODO transform strings into float code

# params
df_path = "../../tmp/working_data/test.csv"
binary_path = "../../tmp/working_data/binaries.csv"

df = pd.read_csv(df_path, sep=";", header=0, encoding="UTF-8")
# df = df.loc[df["EventName"] == "Brazilian Grand Prix"]
binary_df = pd.read_csv(binary_path, sep=";", header=0, encoding="UTF-8").drop(
    "Unnamed: 0", axis=1
)

# process the variables
x = binary_df.iloc[:, :]
x = x.drop("LapTime", axis=1)
y = binary_df["LapTime"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=45
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# créer le modèle
linear_model = LinearRegression()

# entraîner le modèle aux données
linear_model.fit(x_train, y_train)

# évaluer la qualité du modèle sur le subset d'entrainement
y_train_predict = linear_model.predict(x_train)
# racine carrée de l'erreur quadratique moyenne du modèle
rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
r2 = r2_score(y_train, y_train_predict)
# on imprime les résultats de cette évaluation
print("La performance du Modèle pour le set de Training")
print("------------------------------------------------")
print("l'erreur RMSE est {:.3}".format(rmse))
print("le score R2 est {:.3}".format(r2))
print("\n")

# évaluation du modèle pour le set de test
y_test_predict = linear_model.predict(x_test)
# racine carrée de l'erreur quadratique moyenne du modèle
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
# score R carré du modèle
r2_test = r2_score(y_test, y_test_predict)
print("La performance du Modèle pour le set de Test")
print("--------------------------------------------")
print("l'erreur RMSE est {:.3}".format(rmse_test))
print("le score R2 est {:.3}".format(r2_test))
print("\n")

# Get coefficients and corresponding variable names
variable_names = x.columns
# lasso_model = Lasso(alpha=0.03)
# lasso_model.fit(x_train, y_train)

# Get coefficients and corresponding variable names
linear_coefficients = linear_model.coef_

# Print the coefficients
print("These are the effects of your variables on lap times :")
print("------------------------------------------------------")
for variable, coef in zip(variable_names, linear_coefficients):
    print(f"{variable}: {coef:.4f}")


# # Calculate residuals
# residuals = y_test - y_test_predict
# # Create a residual plot
# plt.scatter(y_test_predict, residuals)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.title('Residual Plot')
# plt.xlabel('Predicted Values')
# plt.ylabel('Residuals')
# plt.show()
#
#
# plt.scatter(y_test, y_test_predict)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
# plt.title('Actual vs. Predicted Plot')
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.show()
#
#
# plt.figure(figsize=(10, 6))
# plt.barh(variable_names, abs(lasso_coefficients))
# plt.title('Feature Importance')
# plt.xlabel('Absolute Coefficient Values')
# plt.show()

# SVR polynomial model
svm_model = SVR(kernel="poly")
svm_model.fit(x_train, y_train)

# Evaluate the model on the training set
y_train_pred_svm = svm_model.predict(x_train)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_svm))
r2_train = r2_score(y_train, y_train_pred_svm)

print("Training Performance for SVM:")
print("RMSE:", rmse)
print("R2:", r2_train)

# Evaluate the model on the testing set
y_test_pred_svm = svm_model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_svm))
r2_test = r2_score(y_test, y_test_pred_svm)

print("Testing Performance for SVM:")
print("RMSE:", rmse)
print("R2:", r2_test)

# # perform a grid search
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
# # Create a grid search object
# grid_search = GridSearchCV(svm_model, param_grid, scoring='neg_mean_squared_error')
#
# # Fit the grid search object to the training data
# grid_search.fit(x_train, y_train)
#
# # Print the best parameters
# print(grid_search.best_params_)
