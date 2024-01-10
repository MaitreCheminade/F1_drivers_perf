import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

# params
df_path = "../../tmp/working_data/test.csv"
df = pd.read_csv(df_path, sep=";", header=0, encoding="UTF-8")

# process the variables
# process the variables
x = df[
    [
        "Compound",
        "EventName",
        "TyreLife",
        "TrackStatus",
        "Position",
        "FreshTyre",
        "Driver",
        "Team",
        "TrackStatus",
    ]
]
x = pd.concat(
    [
        x,
        pd.get_dummies(
            x[
                [
                    "Driver",
                    "Compound",
                    "EventName",
                    "TyreLife",
                    "TrackStatus",
                    "Position",
                    "FreshTyre",
                    "Team",
                    "TrackStatus",
                ]
            ],
            drop_first=True,
        ),
    ],
    axis=1,
)
x = x.drop(
    [
        "Driver",
        "Compound",
        "EventName",
        "TyreLife",
        "TrackStatus",
        "Position",
        "FreshTyre",
        "Team",
        "TrackStatus",
    ],
    axis=1,
)

y = df["LapTime"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=45
)

# model
model = SVR(kernel="poly")
c_val = cross_val_score(model, x, y, cv=10)

# Print the results
print(c_val)
