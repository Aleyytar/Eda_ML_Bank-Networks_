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

#Veri tabanı içerisinde kolonları tek tek gezerek içerisinde bulunan nesnel (string vb.) değerleri
#Sayısal Değerlere Dönüştürmek için 
le = LabelEncoder()

for i in df.select_dtypes('object').columns:
    df[i] = le.fit_transform(df[i])

#Data setindeki tüm columları görebilmek için
 # Tüm sütunları göster
pd.set_option('display.max_columns', None) 
 # 10 satır göster
pd.set_option('display.max_rows', 10)     # 10 satır göster
## print(df)
#Datasetini sayısal verilere dönüştürülmüş halini saveledik
df.to_csv("output.csv", index=False)
##print(df.columns)
print(df)

#StandardScaler, veri özelliklerini (örneğin, sütunlardaki değerleri) ortalama değeri 
#0 ve standart sapması 1 olacak şekilde ölçeklendirmeye yarayan bir ölçeklendirme yöntemidir. Bu işlem, veriye aynı ölçekte bakmayı 
#sağlar ve bazı makine öğrenimi algoritmalarının daha iyi performans göstermesine yardımcı olur.
st = StandardScaler()
df["balance"] = st.fit_transform(df[["balance"]])
df["duration"] = st.fit_transform(df[["duration"]])
print(df)