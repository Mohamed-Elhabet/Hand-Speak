from django.db.models.signals import post_save 
from users.models import Profile
from django.contrib.auth.models import User 

def createProfile(sender, instance, created, **kwargs):
    if created:
        user = instance
        Profile.objects.create(user=user)


post_save.connect(createProfile, sender=User)