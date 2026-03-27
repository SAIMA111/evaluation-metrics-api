import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("telco.csv")

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] = df[c].astype(str).str.strip().str.lower()

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df['churn'] = df['churn'].astype(str).str.lower()
df['churn'] = df['churn'].apply(lambda x: 1 if 'yes' in x else 0)

print(df['churn'].value_counts())

numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender','seniorcitizen','partner','dependents','phoneservice',
    'multiplelines','internetservice','onlinesecurity','onlinebackup',
    'deviceprotection','techsupport','streamingtv','streamingmovies',
    'contract','paperlessbilling','paymentmethod',
]

dv = DictVectorizer(sparse=False)

train_dict = df[categorical + numerical].to_dict(orient='records')
X = dv.fit_transform(train_dict)

y = df.churn.values

model = LogisticRegression(max_iter=5000, class_weight='balanced')
model.fit(X, y)

df_processed = df.copy()