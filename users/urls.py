from django.urls import path 
from . import views 

urlpatterns = [
    path('login/', views.login_page, name='login'),
    path('register/', views.register_page, name='register'),
    path('logout/', views.logout_page, name='logout'),
    path('profile/', views.profile, name='profile'),
    path('team/', views.team, name='team'),
    path('update/', views.update_profile, name='update')
]
