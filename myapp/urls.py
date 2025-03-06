from django.urls import path #type: ignore
from . import views  
from django.conf import settings #type: ignore
from django.conf.urls.static import static #type: ignore

urlpatterns = [
  path('', views.home, name='home'),
  path('emotion_details/', views.emotion_details, name='emotion_details'),
  path('emotion_model/', views.emotion_model, name='emotion_model'),
  path('show_result_emo/', views.show_result_emo, name='show_result_emo'),
  path('show_result_num/', views.show_result_num, name='show_result_num'),
  path('number_details/', views.number_details, name='number_details'),
  path('number_model/', views.number_model, name='number_model'),
  path("process_image/", views.process_image, name="process_image"),
  path('user_details/', views.user_details, name='user_details'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)