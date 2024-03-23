
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import xlsxwriter
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_breast_cancer
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, balanced_accuracy_score
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('ignore')

# Veri okuma
df = pd.read_csv("bank.csv", sep=",", encoding='utf-8')

# Veriyi inceleme
print(df.info())
print(df.head())
print(df.describe())
print(df["age"].value_counts().head(15))

# Sütunların türlerini dönüştürme
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Sayısal verilere dönüşüm
df.to_csv("output.csv", index=False)

# Özellik ve etiket ayırma
x = df.drop(['contact', 'day', 'month', 'pdays', 'previous', 'deposit'], axis=1)
y = df["deposit"]

# Veriyi ölçekleme
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Veriyi eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# LazyClassifier kullanarak model metriklerini gösterme
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)

print(models)
