import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ridge_regression, LinearRegression, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.svm import SVR


# params
df_path = "../tmp/working_data/test.csv"
df = pd.read_csv(df_path, sep=";", header=0, encoding="UTF-8")

# process the variables
x = df[["Compound", "EventName", "TyreLife", "TrackStatus", "Position", "FreshTyre", "Driver", "Team", "TrackStatus"]]
x = pd.concat([x, pd.get_dummies(x[["Driver", "Compound", "EventName", "TyreLife", "TrackStatus", "Position", "FreshTyre", "Team", "TrackStatus"]], drop_first=True)], axis=1)
x = x.drop(["Driver", "Compound", "EventName", "TyreLife", "TrackStatus", "Position", "FreshTyre", "Team", "TrackStatus"], axis=1)

y = df["LapTime"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# créer le modèle
n_trees = 150
forest_model = RandomForestRegressor(n_estimators=n_trees)

# entraîner le modèle aux données
forest_model.fit(x_train, y_train)

# évaluer la qualité du modèle sur le subset d'entrainement
y_train_predict = forest_model.predict(x_train)
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
y_test_predict = forest_model.predict(x_test)
# racine carrée de l'erreur quadratique moyenne du modèle
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
# score R carré du modèle
r2_test = r2_score(y_test, y_test_predict)
print("La performance du Modèle pour le set de Test")
print("--------------------------------------------")
print("l'erreur RMSE est {:.3}".format(rmse_test))
print('le score R2 est {:.3}'.format(r2_test))
print("\n")

# Get feature importances
feature_importances = forest_model.feature_importances_

# Sort feature importances by descending order
sorted_idx = np.argsort(feature_importances)[::-1]

# Print feature importances in descending order
print("Feature importances:")
for i in sorted_idx:
    print(f"{x.columns[i]}: {feature_importances[i]:.4f}")

