import json
import base64
import numpy as np 
import pandas as pd 
from PIL import Image 
from io import BytesIO
from django.shortcuts import render
from .data.emotion import use_model as model_emo
from .data.number import used_model as model_num
from .models import User_Details

def addToDB(model, predict, confidence):
  confidenceTostr = str(confidence)
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
  image_data = image_data.split(",")[1]  
  img = Image.open(BytesIO(base64.b64decode(image_data)))

  img = img.convert("L").resize((28, 28))  
  img_array = np.array(img) / 255.0  

  result = model_num.predict(img_array)
  graph = model_num.graph(result['all_predictions'])  
  details = {
    "predict": None,
    "confidence": None,
    "graph": None
  }
  details['predict'] = int(result.get('predict'))
  details['confidence'] = round(float(result.get('confidence')), 4)
  details['graph'] = graph
  addToDB("Digit", details['predict'], details['confidence'])
  return details

def show_result_num(request):
  if request.method == "POST":
    try:
      image_data = request.POST.get("image", "")

      if not image_data:
        return render(request, "error_page.html", {"message": "No image data received"})
      details = {
        "predict": None,
        "confidence": None,
        "graph": None
      }

      details = process_image_function(image_data)

      return render(request, "show_result_num.html", {"details": details})
    except Exception as e:
      return render(request, "error_page.html", {"message": str(e)})

  return render(request, "error_page.html", {"message": "Invalid request"})

def user_details(request):
  data = User_Details.objects.all()
  return render(request, 'user_details.html', {'data': data})
