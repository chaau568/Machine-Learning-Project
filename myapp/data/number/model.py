import tensorflow as tf 
from keras import layers, models 
from keras.callbacks import EarlyStopping 
import numpy as np 
import pandas as pd 

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0 # scaling ข้อมูลด้วยค่าสูงสุด เพื่อให้ข้อมูลอยู่ในช่วง 0-1
X_test = X_test / 255.0 # scaling ข้อมูลด้วยค่าสูงสุด เพื่อให้ข้อมูลอยู่ในช่วง 0-1

# แสดงจำนวนของ dataset
data = {
    "Dataset": ["X_train", "y_train", "X_test", "y_test"],
    "Shape": [X_train.shape, y_train.shape, X_test.shape, y_test.shape],
    "Number of Examples": [X_train.shape[0], y_train.shape[0], X_test.shape[0], y_test.shape[0]],
    "Image Size (Height x Width)": ["28x28", "-", "28x28", "-"]
}
df = pd.DataFrame(data) # รวมเป็น dataframe
# print(df)

# Sequential model เป็นโมเดลแบบเส้นตรง โดยมี 4 layer: 
# 1)Convolutional Layer: ใช้ในการดึงคุณลักษณะของข้อมูล ในที่นี้คือ ภาพ2มิติ 
# โดยกำหนดให้มี 10 Features หรือ Kernel และมีการเพิ่มขอบ padding='same' 
# เพื่อให้การเรียนรู้ของโมเดลเก็บรายละเอียดบริเวณขอบได้ด้วย ทำให้เมื่อผ่าน layer นี้จะได้ข้อมูล 28 * 28 * 10 
# 2)Activation Function: ReLU (Rectified Linear Unit) เพื่อลบค่าลบออกจากฟีเจอร์แมป 
# ทำให้เครือข่ายสามารถเรียนรู้ฟีเจอร์ที่ไม่เป็นเชิงเส้นได้ดีขึ้น
# 3)Pooling Layer: เป็นการลดขนาดข้อ input ที่ได้จาก layer ก่อนหน้าลง(Convolutional Layer) 
# โดยกำหนดให้ลดลง 2 เท่า และ ใช้ pooling แบบ maxpool คือจะดึงเอา Features ที่มีค่ามากที่สุดมาใช้
# 4)Fully Connected Layer: เชื่อมโยงฟีเจอร์ทั้งหมดเข้ากับโครงข่ายแบบ Fully Connected คือเชื่อมโยงทุกๆ 
# node หรือ dense เข้าด้วยกัน ในที่นี้ใช้ 64 dense สำหรับเหตุผลที่ใช้ 64 คือ ถ้าใช้จำนวนที่กำลัง2 เช่น 8 16 32 64 128 
# เป็นตัวเลขที่ลงตัวกับการประมวณผลของคอมพิวเตอร์ ทำให้ได้ประสิธิภาพดีขึ้น และจากจำนวนข้อมูลที่เยอะ (X_train: 50000) 
# จึงควรใช้ dense ที่เยอะพอให้ model เรียนรู้ได้ครอบคลุม ซึ่ง 64 ก็กำลังดีเลย
# Input Layer ไม่ได้รวมใน Sequential model เเต่เป็น Layer ที่จะส่งข้อมูลให้ model ฝึก
# Output Layer ก็ไม่ได้รวมใน Sequential model เช่นกัน แต่เป็นการรับค่าจาก Fully Connected Layer ที่ layer สุดท้ายมาแสดงผล

model = models.Sequential() 
model.add(layers.InputLayer(shape=(28, 28, 1))) 

model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same')) 
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2))) 

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same')) 
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same')) 
model.add(layers.MaxPooling2D((2, 2))) 

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same')) 
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same')) 
model.add(layers.MaxPooling2D((2, 2))) 

model.add(layers.Dropout(0.1)) 
model.add(layers.Flatten()) 

model.add(layers.Dense(64, activation="tanh")) 
model.add(layers.Dense(10, activation="softmax")) 

# model.summary()

model.compile(
  loss="sparse_categorical_crossentropy", # วัดความผิดพลาดระหว่างการทำนายของโมเดลกับค่าจริง
  optimizer="adam", # ช่วยปรับค่าพารามิเตอร์ของโมเดลให้เหมาะสมที่สุด นิยมใช้ 'adam'
  metrics=["accuracy"]) # ใช้ในการติดตามว่าโมเดลทำงานได้ดีแค่ไหนในระหว่างการฝึกและการทดสอบ

early_stopping = EarlyStopping(
  monitor='val_loss', # ติดตาม val_loss
  patience=3, # หยุดการฝึกหลังจาก 3 epochs ที่ val_loss ไม่ดีขึ้น
  restore_best_weights=True # คืนค่าพารามิเตอร์ที่ดีที่สุดที่ val_loss ต่ำสุด
)

# ฝึก model โดยกำหนดให้แบ่งข้อมูลการฝึกออกเป็น 20 ครั้ง epochs=20
model.fit(
  X_train, y_train, epochs=20, 
  validation_data=(X_test, y_test), 
  callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)

# print("Test Accuracy:", accuracy)

model.save('myapp/data/number/model.h5') 
