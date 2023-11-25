from django.urls import path 
from . import views 

urlpatterns = [
    path('translate/<str:id>/', views.translate_video, name='translate-video'),
    path('', views.upload_video, name='upload-video'),
    path('search/', views.search, name='search'),
    path('learn/', views.learn, name='learn'),

]
