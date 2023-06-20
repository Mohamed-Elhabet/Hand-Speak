from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout

from users.forms import CustomUserCreationForm
# Create your views here.


def login_page(request):
    page = 'login'
    if request.user.is_authenticated:
        return redirect('upload-video')
    
    if request.method == 'POST':
        name = request.POST.get('name')
        password = request.POST.get('password')

        try:
            user = User.objects.get(username=name)
        except:
            pass 

        user = authenticate(request, username=user, password=password)
        if user is not None:
            login(request, user)
            return redirect('upload-video')

    context = {
        'page': page
    }
    return render(request, 'users/login.html', context)



def register_page(request):
    page = 'register'
    # form = CustomUserCreationForm()
    if request.method == 'POST':
        name = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        # password2 = request.POST.get('password2')
        # form = CustomUserCreationForm(username=name, email=email, password1=password1, password2=password2)
        user = User.objects.create(username=name, email=email, password=password1)
        login(request, user)
        return redirect('upload-video')
        

    context = {
        'page': page,
        # 'form': form,
    }
    return render(request, 'users/login.html', context)