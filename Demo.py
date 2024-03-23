import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import xlsxwriter
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

#Veri okuma
df = pd.read_csv("bank.csv" , sep=",", encoding='utf-8')

df.info()
print(df.head()) ## Başlık
print(df.describe()) ## Tanım
print(df["age"].value_counts().head(15)) ## Tekrar eden yaş grupları ve tekrar sayıları

sns.histplot(x="age",data=df ,color = 'rosybrown')
##plt.show()

# Yalnızca sayısal verileri içeren sütunları seç
numeric_df = df.select_dtypes(include=[np.number])

# Korelasyon matrisini çiz
sns.heatmap(numeric_df.corr(), annot=True)
##plt.show()

#DataSeti içerisinde Empty Simple sayısı 
df.isnull().sum()/df.shape[0]
## print(IsEmptySimpleCount)

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


x = df.drop([ 'contact', 'day', 'month','pdays','previous','deposit'],axis =1)

y = df["deposit"]
#StandardScaler, veri özelliklerini (örneğin, sütunlardaki değerleri) ortalama değeri 
#0 ve standart sapması 1 olacak şekilde ölçeklendirmeye yarayan bir ölçeklendirme yöntemidir. Bu işlem, veriye aynı ölçekte bakmayı 
#sağlar ve bazı makine öğrenimi algoritmalarının daha iyi performans göstermesine yardımcı olur.
st = StandardScaler()
df["balance"] = st.fit_transform(df[["balance"]])
df["duration"] = st.fit_transform(df[["duration"]])

# x_train: Eğitim setindeki bağımsız değişkenlerin değerleri.
# x_test: Test setindeki bağımsız değişkenlerin değerleri.
# y_train: Eğitim setindeki bağımlı değişkenin değerleri.
# y_test: Test setindeki bağımlı değişkenin değerleri.
x_train , x_test ,y_train ,y_test =train_test_split(x,y ,test_size=0.25 ,random_state= 42)
print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_train.shape)
print(y_test.shape)

def train_evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)  # model eğitme
    predictions = model.predict(x_test)  # test üzerinden tahmin yapabilme

    # metrikler
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    # sonuç çıktısı
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Balanced Accuracy:", balanced_accuracy)

    eval_df = pd.DataFrame([[accuracy, f1, precision, recall, balanced_accuracy]],
                           columns=['accuracy', 'f1_score', 'precision', 'recall', 'balanced_accuracy'])
    return eval_df

# Logistic Regression modeli için sonuçları ekrana yazdırma
lg = LogisticRegression(penalty="l2", C=0.5)
Model1 = train_evaluate_model(lg, x_train, y_train, x_test, y_test)
Model1.index = ['LogisticRegression']

# Decision Tree modeli için sonuçları ekrana yazdırma
decision_tree = DecisionTreeClassifier(max_depth=5, max_features=16)
decision_tree_results = train_evaluate_model(decision_tree, x_train, y_train, x_test, y_test)
decision_tree_results.index = ['DecisionTree']
Model2 = decision_tree_results

print("\nLogistic Regression Results:")
print(Model1)

print("\nDecision Tree Results:")
print(Model2)