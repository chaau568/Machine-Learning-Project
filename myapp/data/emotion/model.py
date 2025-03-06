import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import neattext.functions as nfx 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
import pickle

# โหลดข้อมูล และ เเสดงข้อมูล
data = pd.read_csv("myapp\data\emotion\emotion_dataset.csv")
# print(data.columns) # เช็คหัว column
# print(data['Emotion'].value_counts()) # เช็คว่าแต่ละ label มีความถี่เท่าไหร่
# print("NULL values: ", data['Emotion'].isnull().sum())  # เช็คว่ามีค่าหายไปหรือไม่
# sns.countplot(x='Emotion',data=data)
# plt.show()

# ทำความสะอาดข้อมูล ใช้ .apply เพื่อทำงานที่ละ row ของแต่ละ Text และ เปลี่ยนเป็น Clean_Text
data['Clean_Text'] = data['Text'].apply(nfx.remove_userhandles)
# print(data.columns)
# print(data[['Text', 'Clean_Text']].sample(20)) # เช็คว่าหลังจาก ทำความสะอาดข้อมูลแล้วได้ผลหรือไม่

# เตรียมข้อมูลให้ model
x = data['Clean_Text'] 
y = data['Emotion'] 
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42) 

vectorizer = CountVectorizer()
lr_classifier = LogisticRegression()
pipe_lr = Pipeline(steps=[('cv',vectorizer),('lr',lr_classifier)])
pipe_lr.fit(X_train,y_train)
accuracy = pipe_lr.score(X_test, y_test)
print(f"Accuracy: {accuracy}") #accuracy: 0.6323050392795555

# svm_classifier = SVC(kernel='rbf', C=10)
# pipe_svm = Pipeline(steps=[('cv', vectorizer), ('svc', svm_classifier)])
# pipe_svm.fit(X_train, y_train)
# accuracy = pipe_svm.score(X_test, y_test)
# print(f"Accuracy: {accuracy}") #accuracy: 0.6374784441463882

# rf_classifier = RandomForestClassifier(n_estimators=10)
# pipe_rf = Pipeline(steps=[('cv',vectorizer),('rf', rf_classifier)])
# pipe_rf.fit(X_train,y_train)
# print(pipe_rf.score(X_test,y_test)) #accuracy: 0.5540333397202529

# tfidf_vectorizer = TfidfVectorizer()
# lr2_classifier = LogisticRegression()
# pipe_lr2 = Pipeline(steps=[('tfidf', tfidf_vectorizer), ('lr',lr2_classifier)])
# pipe_lr2.fit(X_train,y_train)
# print(pipe_lr2.score(X_test,y_test)) #sccuracy: 0.619084115730983

# Model ที่ดีที่สุดคือ lr_classifier เพราะมีค่า accuracy ที่มากอันดับ 2 รองจาก svm 
# แต่ svm ใช้ทรัพยากรมากกว่ามาก

# บันทึกโมเดล
with open('myapp\data\emotion\model.pkl', 'wb') as f:
  pickle.dump(pipe_lr, f)
