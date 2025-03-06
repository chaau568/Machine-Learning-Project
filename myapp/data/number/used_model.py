import matplotlib #type: ignore
matplotlib.use('Agg')  # ใช้ non-GUI backend

import io
import base64
import tensorflow as tf #type: ignore
import pandas as pd #type: ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore

def predict(user_input):
    # ตรวจสอบขนาดของ user_input
    if user_input.shape != (28, 28):
        print("Input image must be of size 28x28")
        return
    
    # แปลงข้อมูลจาก (28, 28) เป็น (1, 28, 28, 1) เพื่อให้เหมาะกับโมเดล
    user_input = np.expand_dims(user_input, axis=-1)  # เพิ่มมิติที่ 3 (channels) เพื่อให้มีขนาดเป็น (28, 28, 1)
    user_input = np.expand_dims(user_input, axis=0)   # เพิ่มมิติที่ 0 (batch size) เพื่อให้มีขนาดเป็น (1, 28, 28, 1)
    
    # โหลดโมเดล
    loaded_model = tf.keras.models.load_model('myapp/data/number/model.h5')
    
    # ทำนายผล
    predictions = loaded_model.predict(user_input)[0]  # ได้เป็น array ขนาด (10,)
    
    # 1️⃣ คลาสที่โมเดลทำนาย
    predicted_class = np.argmax(predictions)  

    # 2️⃣ ค่าความมั่นใจของคลาสที่ทำนาย
    confidence = predictions[predicted_class] * 100  

    # 3️⃣ ค่าความมั่นใจของทุกคลาส
    all_confidences = predictions * 100  

    result = {
        'predict': predicted_class,          # Class ที่โมเดลทำนาย
        'confidence': confidence,  # ค่าความมั่นใจของ class นั้น
        'all_predictions': all_confidences  # รายการค่าความมั่นใจของทุกคลาส
    }
    return result
def graph(confidence):
    # เตรียมข้อมูลสำหรับแสดงผล
    classes = np.arange(10)  # หมายเลข 0-9 (10 คลาส)
    confidence_scores = [c * 1 for c in confidence]  # แปลงเป็นเปอร์เซ็นต์

    colors = plt.cm.get_cmap('tab10', 10)(range(10))

    plt.figure(figsize=(6, 6))
    plt.bar(classes, confidence_scores, color=colors)  # สร้าง bar chart
    
    plt.xlabel("Digit Class")
    plt.ylabel("Confidence (%)")
    plt.title("Predicted Probabilities")
    plt.xticks(classes)  # กำหนดให้แกน X เป็น 0-9
    plt.ylim(0, 100)  # กำหนดช่วง Y ให้อยู่ที่ 0-100%

    # แสดงค่าความมั่นใจบนแท่ง
    for i, score in enumerate(confidence_scores):
        plt.text(i, score + 2, f"{score:.2f}%", ha='center', fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    uri = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return uri
