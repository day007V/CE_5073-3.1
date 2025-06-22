import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df['target_class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train.to_numpy(), y_test.to_numpy(), scaler
