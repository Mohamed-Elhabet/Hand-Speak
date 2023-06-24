from django.contrib import admin

from videos.models import LearningVideo, Video

# Register your models here.
admin.site.register(Video)
admin.site.register(LearningVideo)