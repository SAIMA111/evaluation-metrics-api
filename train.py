import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def train_model():
    df = pd.read_csv("telco.csv")

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'yes').astype(int)

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    del df_train['churn']
    del df_val['churn']

    numerical = ['tenure', 'monthlycharges', 'totalcharges']

    categorical = [
        'gender','seniorcitizen','partner','dependents','phoneservice',
        'multiplelines','internetservice','onlinesecurity','onlinebackup',
        'deviceprotection','techsupport','streamingtv','streamingmovies',
        'contract','paperlessbilling','paymentmethod',
    ]

    dv = DictVectorizer(sparse=False)

    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, dv, df_val, y_val