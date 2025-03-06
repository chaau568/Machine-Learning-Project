import matplotlib #type: ignore
matplotlib.use('Agg')  # ใช้ non-GUI backend

import io
import os
import base64
import pickle
import numpy as np #type: ignore
import pandas as pd #type: ignore
import matplotlib.pyplot as plt #type: ignore

def predict(user_input):
  file_path = os.path.join('myapp', 'data', 'emotion', 'model.pkl')
  if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
      loaded_model = pickle.load(f)
    print("Model loaded successfully!")
  else:
    print(f"Error: File '{file_path}' not found!")

  classes = loaded_model.classes_

  if not user_input:
    return "Error: Not found input"
  
  y_pred_prob = loaded_model.predict_proba([user_input])[0]
  max_class_index = np.argmax(y_pred_prob)
  confidence = y_pred_prob[max_class_index]
  predict_result = classes[max_class_index]
  result = {
    'predict': predict_result,
    'confidence': f'{(confidence * 100):.6f} %'
  }
  return result

def graph(user_input):
  file_path = os.path.join('myapp', 'data', 'emotion', 'model.pkl')
  if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
      loaded_model = pickle.load(f)
    print("Model loaded successfully!")
  else:
    print(f"Error: File '{file_path}' not found!")

  classes = loaded_model.classes_

  if not user_input:
    return None
  
  y_pred_prob = loaded_model.predict_proba([user_input])[0]
  if len(y_pred_prob) == 0:
    return None  
        
  data_sorted = pd.DataFrame({
    'class': classes,
    'probability': y_pred_prob * 100
  })

  plt.figure(figsize=(6, 6))
  bars = plt.bar(
    data_sorted['class'], data_sorted['probability'],
    color=plt.cm.Paired(np.linspace(0, 1, len(classes)))
  )

  plt.title('Predicted Probabilities')
  plt.xlabel('Emotion Class')
  plt.ylabel('Confidence (%)')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.ylim(0, 100) 

  for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight')
  plt.close()
  buf.seek(0)
  uri = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')
  buf.close()

  return uri