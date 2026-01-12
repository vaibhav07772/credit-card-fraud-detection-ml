import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("credit_card_dataset.csv")

X = df.drop("IsFraud", axis=1)
y = df["IsFraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBClassifier()
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print("Accuracy:", acc)

joblib.dump(model, "models/model_v1.pkl")