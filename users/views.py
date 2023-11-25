from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout

from users.forms import CustomUserCreationForm, UserProfileUpdateForm, UserUpdateForm
from users.models import Profile
# Create your views here.


def login_page(request):
    page = 'login'
    error = None 
    if request.user.is_authenticated:
        return redirect('upload-video')
    
    try:
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
    except:
        error = 'Username Or Password Incorrect'
        pass 

    context = {
        'page': page,
        'error': error
    }
    return render(request, 'users/login.html', context)




def register_page(request):
    page = 'register'
    form = CustomUserCreationForm()
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('upload-video')
        

    context = {
        'page': page,
        'form': form,
    }
    return render(request, 'users/login.html', context)





def logout_page(request):
    logout(request)
    return redirect('login')




def profile(request):
    user = request.user
    profile = Profile.objects.get(user=user)
    context = {
        'user': user,
        'profile': profile
    }
    return render(request, 'users/profile.html', context)



def update_profile(request):
    user = request.user
    userForm = UserUpdateForm(instance=request.user)
    profileForm = UserProfileUpdateForm(instance=request.user.profile)

    if request.method == 'POST':
        userForm = UserUpdateForm(request.POST, instance=user)
        profileForm = UserProfileUpdateForm(request.POST, request.FILES, instance=user.profile)

        if userForm.is_valid() and profileForm.is_valid():
            userForm.save()
            profileForm.save()
            return redirect('profile')
        
    context = {
        'userForm': userForm,
        'profileForm': profileForm
    }
    return render(request, 'users/update.html', context)



def team(request):
    return render(request, 'users/team.html')