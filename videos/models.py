from django.db import models
from django.contrib.auth.models import User 

# Create your models here.

class Video(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.FileField(upload_to='')
    translate = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.translate



class LearningVideo(models.Model):
    translate = models.CharField(max_length=250)
    video = models.FileField(upload_to='')

    def __str__(self):
        return self.translate