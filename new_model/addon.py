import numpy as np 
from .dtw import calculate_mean_pixel_value,frame_num
from .loading import df



def addons(video_path):
    y = calculate_mean_pixel_value(video_path)
    x = frame_num(video_path)
   
    info = df.loc[(df["frames"] == x) & (df["mean"] == y), "sign"].values

    return info[0]