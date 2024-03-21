import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, cv, Pool   
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, balanced_accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

#Read the Database
df = pd.read_csv("bank.csv" , sep=",", encoding='utf-8')

df.info()
print(df.head()) ## Başlık
print(df.describe()) ## Tanım
print(df["age"].value_counts().head(15)) ## Tekrar eden yaş grupları ve tekrar sayıları

sns.histplot(x="age",data=df ,color = 'rosybrown')
##plt.show()
# Veritabanını Oku
df = pd.read_csv("bank.csv", sep=",", encoding='utf-8')

# Yalnızca sayısal verileri içeren sütunları seç
numeric_df = df.select_dtypes(include=[np.number])

# Korelasyon matrisini çiz
sns.heatmap(numeric_df.corr(), annot=True)
##plt.show()

#DataSeti içerisinde Empty Simple sayısı 
IsEmptySimpleCount = df.isnull().sum()/df.shape[0]

print(IsEmptySimpleCount)