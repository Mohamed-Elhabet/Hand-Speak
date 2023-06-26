from django.shortcuts import render, redirect
from videos.forms import VideoForm

from videos.models import LearningVideo, Video
from django.contrib.auth.decorators import login_required
from .main_main import prediction

# Create your views here.

'''
def upload_video(request):
    print(request.user)
    form = VideoForm()
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid:
            video = form.save(commit=False)
            video.user = request.user 
            video.save()
            id_video = video.id 
        return redirect('translate-video', id_video)

    return render(request, 'videos/upload.html', {'form': form})
'''

@login_required(login_url='login')
def translate_video(request, id):
    try:
        video = Video.objects.get(id=id)
    except:
        video = None 

    context = {
        'vid': video
    }

    return render(request, 'videos/translate.html', context)





@login_required(login_url='login')
def search(request):
    q = request.GET.get('search') if request.GET.get('search') != None else ""
    video = LearningVideo.objects.filter(translate__iexact=q).first()

    print(video)

    context = {
        'vid': video
    }
    return render(request, 'videos/translate.html', context)




@login_required(login_url='login')
def learn(request):
    videos = LearningVideo.objects.all()
    context = {
        'videos': videos
    }
    return render(request, 'videos/all_videos.html', context)



@login_required(login_url='login')
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




















############# AI ##############################


  
# import tensorflow as tf 
# import cv2 
# import numpy as np
# from tensorflow import keras
# import mediapipe as mp

# import pandas as pd
# import json
# import os
# from multiprocessing import cpu_count

# ROWS_PER_FRAME = 543
# MAX_LEN = 384
# CROP_LEN = MAX_LEN
# NUM_CLASSES  = 250
# PAD = -100.
# NOSE=[
#     1,2,98,327
# ]
# LNOSE = [98]
# RNOSE = [327]
# LIP = [ 0, 
#     61, 185, 40, 39, 37, 267, 269, 270, 409,
#     291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
#     78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
#     95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
# ]
# LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
# RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

# POSE = [500, 502, 504, 501, 503, 505, 512, 513]
# LPOSE = [513,505,503,501]
# RPOSE = [512,504,502,500]

# REYE = [
#     33, 7, 163, 144, 145, 153, 154, 155, 133,
#     246, 161, 160, 159, 158, 157, 173,
# ]
# LEYE = [
#     263, 249, 390, 373, 374, 380, 381, 382, 362,
#     466, 388, 387, 386, 385, 384, 398,
# ]

# LHAND = np.arange(468, 489).tolist()
# RHAND = np.arange(522, 543).tolist()

# POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE #+POSE

# NUM_NODES = len(POINT_LANDMARKS)
# CHANNELS = 6*NUM_NODES



# def tf_nan_mean(x, axis=0, keepdims=False):
#     return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

# def tf_nan_std(x, center=None, axis=0, keepdims=False):
#     if center is None:
#         center = tf_nan_mean(x, axis=axis,  keepdims=True)
#     d = x - center
#     return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

# class Preprocess(tf.keras.layers.Layer):
#     def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
#         super().__init__(**kwargs)
#         self.max_len = max_len
#         self.point_landmarks = point_landmarks

#     def call(self, inputs):
#         if tf.rank(inputs) == 3:
#             x = inputs[None,...]
#         else:
#             x = inputs
        
#         mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1,2], keepdims=True)
#         mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5,x.dtype), mean)
#         x = tf.gather(x, self.point_landmarks, axis=2) #N,T,P,C
#         std = tf_nan_std(x, center=mean, axis=[1,2], keepdims=True)
        
#         x = (x - mean)/std

#         if self.max_len is not None:
#             x = x[:,:self.max_len]
#         length = tf.shape(x)[1]
#         x = x[...,:2]

#         dx = tf.cond(tf.shape(x)[1]>1,lambda:tf.pad(x[:,1:] - x[:,:-1], [[0,0],[0,1],[0,0],[0,0]]),lambda:tf.zeros_like(x))

#         dx2 = tf.cond(tf.shape(x)[1]>2,lambda:tf.pad(x[:,2:] - x[:,:-2], [[0,0],[0,2],[0,0],[0,0]]),lambda:tf.zeros_like(x))

#         x = tf.concat([
#             tf.reshape(x, (-1,length,2*len(self.point_landmarks))),
#             tf.reshape(dx, (-1,length,2*len(self.point_landmarks))),
#             tf.reshape(dx2, (-1,length,2*len(self.point_landmarks))),
#         ], axis = -1)
        
#         x = tf.where(tf.math.is_nan(x),tf.constant(0.,x.dtype),x)
        
#         return x
    




# class ECA(tf.keras.layers.Layer):
#     def __init__(self, kernel_size=5, **kwargs):
#         super().__init__(**kwargs)
#         self.supports_masking = True
#         self.kernel_size = kernel_size
#         self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

#     def call(self, inputs, mask=None):
#         nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
#         nn = tf.expand_dims(nn, -1)
#         nn = self.conv(nn)
#         nn = tf.squeeze(nn, -1)
#         nn = tf.nn.sigmoid(nn)
#         nn = nn[:,None,:]
#         return inputs * nn

# class LateDropout(tf.keras.layers.Layer):
#     def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
#         super().__init__(**kwargs)
#         self.supports_masking = True
#         self.rate = rate
#         self.start_step = start_step
#         self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)
      
#     def build(self, input_shape):
#         super().build(input_shape)
#         agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
#         self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

#     def call(self, inputs, training=False):
#         x = tf.cond(self._train_counter < self.start_step, lambda:inputs, lambda:self.dropout(inputs, training=training))
#         if training:
#             self._train_counter.assign_add(1)
#         return x

# class CausalDWConv1D(tf.keras.layers.Layer):
#     def __init__(self, 
#         kernel_size=17,
#         dilation_rate=1,
#         use_bias=False,
#         depthwise_initializer='glorot_uniform',
#         name='', **kwargs):
#         super().__init__(name=name,**kwargs)
#         self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(kernel_size-1),0),name=name + '_pad')
#         self.dw_conv = tf.keras.layers.DepthwiseConv1D(
#                             kernel_size,
#                             strides=1,
#                             dilation_rate=dilation_rate,
#                             padding='valid',
#                             use_bias=use_bias,
#                             depthwise_initializer=depthwise_initializer,
#                             name=name + '_dwconv')
#         self.supports_masking = True
        
#     def call(self, inputs):
#         x = self.causal_pad(inputs)
#         x = self.dw_conv(x)
#         return x

# def Conv1DBlock(channel_size,
#           kernel_size,
#           dilation_rate=1,
#           drop_rate=0.0,
#           expand_ratio=2,
#           se_ratio=0.25,
#           activation='swish',
#           name=None):
#     '''
#     efficient conv1d block, @hoyso48
#     '''
#     if name is None:
#         name = str(tf.keras.backend.get_uid("mbblock"))
#     # Expansion phase
#     def apply(inputs):
#         channels_in = tf.keras.backend.int_shape(inputs)[-1]
#         channels_expand = channels_in * expand_ratio

#         skip = inputs

#         x = tf.keras.layers.Dense(
#             channels_expand,
#             use_bias=True,
#             activation=activation,
#             name=name + '_expand_conv')(inputs)

#         # Depthwise Convolution
#         x = CausalDWConv1D(kernel_size,
#             dilation_rate=dilation_rate,
#             use_bias=False,
#             name=name + '_dwconv')(x)

#         x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

#         x  = ECA()(x)

#         x = tf.keras.layers.Dense(
#             channel_size,
#             use_bias=True,
#             name=name + '_project_conv')(x)

#         if drop_rate > 0:
#             x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1), name=name + '_drop')(x)

#         if (channels_in == channel_size):
#             x = tf.keras.layers.add([x, skip], name=name + '_add')
#         return x

#     return apply





# class MultiHeadSelfAttention(tf.keras.layers.Layer):
#     def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
#         super().__init__(**kwargs)
#         self.dim = dim
#         self.scale = self.dim ** -0.5
#         self.num_heads = num_heads
#         self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
#         self.drop1 = tf.keras.layers.Dropout(dropout)
#         self.proj = tf.keras.layers.Dense(dim, use_bias=False)
#         self.supports_masking = True

#     def call(self, inputs, mask=None):
#         qkv = self.qkv(inputs)
#         qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
#         q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

#         attn = tf.matmul(q, k, transpose_b=True) * self.scale

#         if mask is not None:
#             mask = mask[:, None, None, :]

#         attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
#         attn = self.drop1(attn)

#         x = attn @ v
#         x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
#         x = self.proj(x)
#         return x


# def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
#     def apply(inputs):
#         x = inputs
#         x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
#         x = MultiHeadSelfAttention(dim=dim,num_heads=num_heads,dropout=attn_dropout)(x)
#         x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
#         x = tf.keras.layers.Add()([inputs, x])
#         attn_out = x

#         x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
#         x = tf.keras.layers.Dense(dim*expand, use_bias=False, activation=activation)(x)
#         x = tf.keras.layers.Dense(dim, use_bias=False)(x)
#         x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
#         x = tf.keras.layers.Add()([attn_out, x])
#         return x
#     return apply






# def get_model(max_len=MAX_LEN, dropout_step=0, dim=192):
#     inp = tf.keras.Input((max_len,CHANNELS))
#     #x = tf.keras.layers.Masking(mask_value=PAD,input_shape=(max_len,CHANNELS))(inp) #we don't need masking layer with inference
#     x = inp
#     ksize = 17
#     x = tf.keras.layers.Dense(dim, use_bias=False,name='stem_conv')(x)
#     x = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)

#     x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#     x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#     x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#     x = TransformerBlock(dim,expand=2)(x)

#     x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#     x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#     x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#     x = TransformerBlock(dim,expand=2)(x)

#     if dim == 384: #for the 4x sized model
#         x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#         x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#         x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#         x = TransformerBlock(dim,expand=2)(x)

#         x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#         x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#         x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#         x = TransformerBlock(dim,expand=2)(x)

#     x = tf.keras.layers.Dense(dim*2,activation=None,name='top_conv')(x)
#     x = tf.keras.layers.GlobalAveragePooling1D()(x)
#     x = LateDropout(0.8, start_step=dropout_step)(x)
#     x = tf.keras.layers.Dense(NUM_CLASSES,name='classifier')(x)
#     return tf.keras.Model(inp, x)





# models_path = [
#               'islr-fp16-192-8-seed42-foldall-last.h5', #comment out other weights to check single model score
#                'islr-fp16-192-8-seed43-foldall-last.h5',
#                'islr-fp16-192-8-seed44-foldall-last.h5',
#                'islr-fp16-192-8-seed45-foldall-last.h5',
#               ]
# models = [get_model() for _ in models_path]
# for model,path in zip(models,models_path):
#     model.load_weights(path)





# class inference(tf.Module):
#     """
#     TensorFlow Lite model that takes input tensors and applies:
#         – a preprocessing model
#         – the ISLR model 
#     """

#     def __init__(self, islr_models):
#         """
#         Initializes the TFLiteModel with the specified preprocessing model and ISLR model.
#         """
#         super(inference, self).__init__()

#         # Load the feature generation and main models
#         self.prep_inputs = Preprocess()
#         self.islr_models   = islr_models
    
#     @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
#     def __call__(self, inputs):
#         """
#         Applies the feature generation model and main model to the input tensors.

#         Args:
#             inputs: Input tensor with shape [batch_size, 543, 3].

#         Returns:
#             A dictionary with a single key 'outputs' and corresponding output tensor.
#         """



#         x = self.prep_inputs(tf.cast(inputs, dtype=tf.float32))
#         outputs = [model(x) for model in self.islr_models]
#         outputs = tf.keras.layers.Average()(outputs)[0]
#         return {'outputs': outputs}
    




# final_model = inference(models)




# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# def mediapipe_detection(image,model):
#     image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image ,cv2.COLOR_RGB2BGR)
#     return image,results


    
# def adjacting_landmarks(results):
#     pose = np.array([[res.x,res.y,res.z] for res in results.pose_landmarks.landmark ]).flatten() if results.pose_landmarks else np.zeros(132)
#     face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark ]).flatten() if results.face_landmarks else np.zeros(1371)
#     lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark ]).flatten() if results.left_hand_landmarks else np.zeros(63)
#     rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark ]).flatten() if results.right_hand_landmarks else np.zeros(63)
#     return np.concatenate([face,lh,pose,rh])










# # def ai_video(request):
# #     vid = ''
# #     prediction = ''
# #     # vid = VidStream.objects.get(id=16)
# #     # vid = VidStream.objects.first()
# #     vid = Video.objects.last()
    
# #     with mp_holistic.Holistic(min_detection_confidence=0.5 ,min_tracking_confidence=0.5) as holistic:
# #         # cap = cv2.VideoCapture("03008.mp4")
# #         cap = cv2.VideoCapture(vid.video.path)
# #         temp = []
# #         while cap.isOpened():
# #             ret,frame = cap.read()
# #             if ret:
# #                 frame,results = mediapipe_detection(frame,holistic)
# #                 results = adjacting_landmarks(results)
# #                 temp.append(results.reshape((543,3)))
# #             else:
# #                 break
# #         cap.release()


# #     data = np.stack(temp ,axis=0).astype(np.float32)
# #     print('SHAPE : ', data.shape)

# #     prediction = np.argmax(final_model(data)["outputs"])

# #     print('ARGMAX : ', prediction)
    

# #     context = {
# #         'video': vid,
# #         # 'prediction': prediction
# #         }
    

# #     return render(request, 'videos/upload.html', context)







# import json 

# # Load the JSON object from file
# with open('sign_to_prediction_index_map.json', 'r') as f:
#     json_data = json.load(f)

# # Create a new dictionary with the keys and values swapped
# new_dict = {str(value): key for key, value in json_data.items()}

# # Print the new dictionary
# # print(new_dict)






# @login_required(login_url='login')
# def upload_video(request):
#     user = request.user
#     print('user ......... ', user)
#     form = VideoForm()
#     if request.method == 'POST':
#         form = VideoForm(request.POST, request.FILES)
#         if form.is_valid():
#             video = form.save(commit=False)
#             video.user = request.user
#             video.save()
#             video_id = video.id
#             # video.save()
            
#             try:
#                 with mp_holistic.Holistic(min_detection_confidence=0.5 ,min_tracking_confidence=0.5) as holistic:
#                     # cap = cv2.VideoCapture("03008.mp4")
#                     cap = cv2.VideoCapture(video.video.path)
#                     temp = []
#                     while cap.isOpened():
#                         ret,frame = cap.read()
#                         if ret:
#                             frame,results = mediapipe_detection(frame,holistic)
#                             results = adjacting_landmarks(results)
#                             temp.append(results.reshape((543,3)))
#                         else:
#                             break
#                     cap.release()


#                 data = np.stack(temp ,axis=0).astype(np.float32)
#                 print('SHAPE : ', data.shape)

#                 prediction = np.argmax(final_model(data)["outputs"])

#                 # with open("sign_to_prediction_index_map.json" ,"r") as file:
#                 #     index_class = json.load(file)

#                 #     # word = index_class[prediction]
#                 #     print(index_class)


#                 # video.description = prediction
#                 video.translate = new_dict[str(prediction)]
#             except:
#                 video.translate = 'Not Known'
#                 print('ARGMAX : ', prediction)
            

#             video.save()




#             return redirect('translate-video', video_id)

#     context = {
#         'form': form,
        
#     }

#     # return render(request, 'stream/post-video.html', context)
#     return render(request, 'videos/upload.html', context)

