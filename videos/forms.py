from django import forms

from videos.models import Video 

class VideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ['video']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['video'].label = False 