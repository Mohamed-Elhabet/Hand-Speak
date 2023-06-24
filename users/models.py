from django.db import models
from django.contrib.auth.models import User 

# Create your models here.


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    photo = models.FileField(null=True, blank=True, default='profiles/user-default.png')
    social_facebook = models.CharField(max_length=300, blank=True, null=True)
    social_linkedin = models.CharField(max_length=300, blank=True, null=True)
    social_github = models.CharField(max_length=300, blank=True, null=True)

    def __str__(self):
        return f'{self.user.username}'
