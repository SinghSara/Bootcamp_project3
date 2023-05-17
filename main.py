import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn import preprocessing

df=pd.read_csv('customer_churn.csv')
ls=[]
for col in df.columns:
    if df[col].nunique() < 6 :
        df[col]=df[col].astype('category')
        ls=ls+[col]
ls.remove('Churn')
pickle.dump(ls, open('cat_col.pkl','wb'))
df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].median())
for col in df.columns:
    if df[col].dtype=='category':
        #df[col] = df[col].cat.codes
        
        label_encoder = preprocessing.LabelEncoder()
        

        df[col]=label_encoder.fit_transform(df[col])
        df[col]=df[col].astype('category')
        filename=col+".pkl"
        pickle.dump(label_encoder, open(filename,'wb'))
df.drop(['customerID'],axis=1,inplace=True)
for col in df.columns:
    if df[col].dtype!='category':
        minmax = MinMaxScaler()
        df[col]=minmax.fit_transform(df[[col]])
pickle.dump(minmax, open('minmax.pkl','wb'))
X,Y=df.loc[:, ~df.columns.isin(['Churn'])],df.loc[:, df.columns.isin(['Churn'])]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33,stratify=Y, random_state=42)
model=LogisticRegression(C=1,penalty='l2',solver='newton-cg')
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print(classification_report(Y_test, y_pred))
pickle.dump(model, open('model.pkl','wb'))

