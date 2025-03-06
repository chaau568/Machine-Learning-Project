import json
import base64
import numpy as np #type: ignore
import pandas as pd #type: ignore
from PIL import Image #type: ignore
from io import BytesIO
from django.http import JsonResponse #type: ignore
from django.views.decorators.csrf import csrf_exempt #type: ignore
from django.shortcuts import render, redirect #type: ignore
from .data.emotion import use_model as model_emo
from .data.number import used_model as model_num
from .models import User_Details

latest_result = {
    "predict": None,
    "confidence": None,
    "graph": None
}
check = 0

def addToDB(model, predict, confidence):
  confidenceTostr = str(confidence)
  print(confidence, confidenceTostr)
  print(type(confidenceTostr))
  User_Details.objects.create(
    model = model,
    predict = predict,
    confidence = confidenceTostr
  )

def home(request):
  return render(request, 'home.html')

def emotion_details(request):
  data = pd.read_csv('myapp/data/details_data/class_details.csv')
  class_details = data[['Emotion', 'Count']]
  class_details_dict = class_details.to_dict(orient='records')
  img_path = 'myapp/data/details_data/accuracy_loss_graph.png'
  with open(img_path, "rb") as img_file:
    img_data = base64.b64encode(img_file.read()).decode('utf-8')
  context = {
    'class_details': class_details_dict,  
    'model_accuracy': img_data
  }
  
  return render(request, 'emotion_details.html', context)

def emotion_model(request):
  return render(request, 'emotion_model.html')

def show_result_emo(request):
  details = {
    'text': None,
    'predict': None,
    'confidence': None,
    'graph': None
  }
  if request.method == "POST":
    user_input = request.POST.get('user_input', '')
    details['text'] = user_input

    result = model_emo.predict(user_input)
    if isinstance(result, dict):
      details['predict'] = result.get('predict', None)
      details['confidence'] = result.get('confidence', None)
    details['graph'] = model_emo.graph(user_input)
    addToDB("emotion", details['predict'], details['confidence'])
  return render(request, 'show_result_emo.html', {"details": details})

def number_details(request):
  img_path = 'myapp/data/details_data/accuracy_loss_plot.png'
  with open(img_path, "rb") as img_file:
    accuracy_loss_plot = base64.b64encode(img_file.read()).decode('utf-8')

  class_count_path = pd.read_csv('myapp/data/details_data/class_counts.csv')
  # แสดงชื่อคอลัมน์
  print(class_count_path.columns)
  class_details = class_count_path[['Class', 'Train', 'Test']]
  class_count = class_details.to_dict(orient='records')

  context = {
    'class_details': class_count,
    'accuracy_loss_plot': accuracy_loss_plot,
  }
  return render(request, 'number_details.html', context)

def number_model(request):
  return render(request, 'number_model.html')

def process_image_function(image_data):
  image_data = image_data.split(",")[1]  # ตัด "data:image/png;base64,"
  img = Image.open(BytesIO(base64.b64decode(image_data)))

  img = img.convert("L").resize((28, 28))  # แปลงเป็นขาวดำ + ปรับขนาด 28x28
  img_array = np.array(img) / 255.0  # Normalize เป็น 0-1

  # ส่งภาพไปยังโมเดลของคุณเพื่อทำนาย
  result = model_num.predict(img_array)

  # เพิ่มค่ากราฟ (ผลลัพธ์ของการทำนายแต่ละคลาส) 
  graph = model_num.graph(result['all_predictions'])  # คุณสามารถดึงกราฟได้จากผลลัพธ์

  # อัพเดตตัวตรวจสอบค่าเก่า
  global check
  check = 1
  # บันทึกผลลัพธ์ไว้
  global latest_result
  latest_result['predict'] = int(result.get('predict'))
  latest_result['confidence'] = round(float(result.get('confidence')), 4)
  latest_result['graph'] = graph
  addToDB("Digit", latest_result['predict'], latest_result['confidence'])
      
@csrf_exempt
def process_image(request):
  if request.method == "POST":
    try:
      data = json.loads(request.body)
      image_data = data.get("image", "")

      if not image_data:
        return JsonResponse({"error": "No image data"}, status=400)

      process_image_function(image_data)

      return JsonResponse({"message": "Image processed successfully!"})
    except Exception as e:
      return JsonResponse({"error": str(e)}, status=500)

  return JsonResponse({"error": "Invalid request"}, status=400)

def show_result_num(request):
  global check
  if check == 0:
    global latest_result
    latest_result['predict'] = None
    latest_result['confidence'] = None
    latest_result['graph'] = None
  check = 0
  return render(request, 'show_result_num.html', {'details': latest_result})

def user_details(request):
  data = User_Details.objects.all()
  return render(request, 'user_details.html', {'data': data})