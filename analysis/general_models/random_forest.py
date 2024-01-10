import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.svm import SVR


# params
df_path = "../../tmp/working_data/filtered.csv"
binary_path = "../../tmp/working_data/binaries.csv"
df = pd.read_csv(df_path, sep=";", header=0, encoding="UTF-8")
binary_df = pd.read_csv(binary_path, sep=";", header=0, encoding="UTF-8").drop(
    "Unnamed: 0", axis=1
)
track_list = [
    "EventName_Austrian Grand Prix",
    "EventName_Belgian Grand Prix",
    "EventName_Singapore Grand Prix",
    "EventName_Azerbaijan Grand Prix",
    "EventName_Sakhir Grand Prix",
    "EventName_Brazilian Grand Prix",
    "EventName_United States Grand Prix",
    "EventName_Russian Grand Prix",
    "EventName_Bahrain Grand Prix",
    "EventName_French Grand Prix",
    "EventName_Chinese Grand Prix",
    "EventName_Japanese Grand Prix",
    "EventName_British Grand Prix",
    "EventName_Portuguese Grand Prix",
    "EventName_Hungarian Grand Prix",
    "EventName_Mexican Grand Prix",
    "EventName_Emilia Romagna Grand Prix",
    "EventName_German Grand Prix",
    "EventName_Spanish Grand Prix",
    "EventName_Italian Grand Prix",
    "EventName_Turkish Grand Prix",
    "EventName_Qatar Grand Prix",
    "EventName_Australian Grand Prix",
    "EventName_Eifel Grand Prix",
    "EventName_Tuscan Grand Prix",
    "EventName_Dutch Grand Prix",
    "EventName_Canadian Grand Prix",
    "EventName_Monaco Grand Prix",
    "EventName_Miami Grand Prix",
    "EventName_Saudi Arabian Grand Prix",
]

# process the variables
x = binary_df.iloc[:, :]
x = x.drop("LapTime", axis=1)
y = binary_df["LapTime"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.22, random_state=24
)
variable_names = x.columns


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# créer le modèle
n_trees = 150
forest_model = RandomForestRegressor(n_estimators=n_trees, random_state=56)

# entraîner le modèle aux données
forest_model.fit(x_train, y_train)

y_train_predict = forest_model.predict(x_train)
y_test_predict = forest_model.predict(x_test)


def evaluate_forest_reg_performance(
    y_training_set, y_training_prediction, y_testing_set, y_testing_prediction
):
    """Evaluate performance of model prediction for training and testing sets,
    benchmarks are RMSE and R2 score
    returns prints"""
    # evaluating performance for training set
    rmse = np.sqrt(mean_squared_error(y_training_set, y_training_prediction))
    r2 = r2_score(y_training_set, y_training_prediction)
    print("La performance du Modèle pour le set de Training")
    print("------------------------------------------------")
    print("l'erreur RMSE est {:.3}".format(rmse))
    print("le score R2 est {:.3}".format(r2))
    print("\n")

    # evaluating performance for testing set
    rmse_test = np.sqrt(mean_squared_error(y_testing_set, y_testing_prediction))
    r2_test = r2_score(y_testing_set, y_testing_prediction)
    print("La performance du Modèle pour le set de Test")
    print("--------------------------------------------")
    print("l'erreur RMSE est {:.3}".format(rmse_test))
    print("le score R2 est {:.3}".format(r2_test))
    print("\n")


evaluate_forest_reg_performance(y_train, y_train_predict, y_test, y_test_predict)
# Get feature importances
feature_importances = forest_model.feature_importances_
# Sort feature importances by descending order
sorted_idx = np.argsort(feature_importances)[::-1]
# Print feature importances in descending order
print("Feature importances:")
for i in sorted_idx:
    print(f"{x.columns[i]}: {feature_importances[i]:.4f}")
