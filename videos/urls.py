from django.urls import path 
from . import views 

urlpatterns = [
    # path('', views.upload_video, name='upload-video'),
    path('translate/<str:id>/', views.translate_video, name='translate-video'),
    # path('live/', views.ai_video, name='live'),

    # path('video-translate-ai/', views.VideoCreateView, name='ai-translate')
    # path('', views.VideoCreateView, name='upload-video')
    path('', views.upload_video, name='upload-video')
]
