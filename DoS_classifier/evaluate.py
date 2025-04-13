# evaluate.py

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def evaluate_model(data_path, model_path):
    df = pd.read_csv(data_path)

    X = df.drop('Label', axis=1)
    y = df['Label']

    model = joblib.load(model_path)
    scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
    X = scaler.transform(X)

    y_pred = model.predict(X)

    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))
