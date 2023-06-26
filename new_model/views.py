from django.shortcuts import redirect, render

from videos.forms import VideoForm
from .main_main import prediction



# video_name = 'test_videos/backpack.mp4'

# if __name__ == "__main__":
#     prediction(video_name)


# Create your views here.

# def new_model_predict(request):
#     predict_label = prediction(video_name)

#     context = {
#         'prediction': predict_label 
#     }
#     return render(request, 'new_model/1.html', context)




def upload_video(request):
    user = request.user 
    print('user ..... ', user)
    form = VideoForm()
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save(commit=False)
            video.user = user 
            video.save()
            video_id = video.id 
            print('path : ', video.video.path)
            print('url : ', video.video.url)
            
            predict_label = prediction(video.video.path)
            # predict_label = prediction(video.video.url)
            print("LABEL : ", predict_label)

            video.translate = predict_label 
            video.save()
            return redirect('translate-video', video_id)
    
    context = {
        'form': form 
    }
    return render(request, 'videos/upload.html', context)

