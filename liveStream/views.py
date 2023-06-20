from django.shortcuts import render, redirect
from tensorflow.keras.models import Sequential ,Model
import numpy as np 
import pandas as pd
import os
import mediapipe as mp
import tensorflow as tf 
import cv2 

from tensorflow.keras.layers import LSTM, Dense
from tensorflow import keras
from django.contrib.auth.decorators import login_required

# Create your views here.


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
def draw_styled_landmarks(image, results):
    # Draw face connections
    
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])



Model = tf.keras.models.load_model("SLI.h5")

actions = ['age', 'book', 'call', 'car', 'day', 'egypt', 'english', 'enjoy',
       'every', 'excuse', 'football', 'forget', 'fun', 'good', 'hate',
       'have', 'hello', 'help', 'holiday', 'iam', 'love', 'meet', 'month',
       'morning', 'my', 'na', 'name', 'nice', 'no', 'not', 'number',
       'okay', 'picture', 'play', 'read', 'ride', 'run', 'sorry', 'speak',
       'sport', 'take', 'thanks', 'time', 'today', 'understand', 'what',
       'when', 'where', 'year', 'yes', 'you', 'your']




@login_required(login_url='login')
def home(request):
    sequence = []
    sentence = []
    threshold = 0.8

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        predicted_actions = []

        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            # sequence.insert(0,keypoints)
            # sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-20:]

            if len(sequence) == 20:
                res = Model.predict(np.expand_dims(sequence, axis=0))[0]
                num = int(np.argmax(res))

                # print(actions[np.argmax(res)])

                predicted_action = actions[np.argmax(res)]
                predicted_actions.append(predicted_action)
                print(predicted_action)

            # 3. Viz logic
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[num] != sentence[-1]:
                            sentence.append(actions[num])
                    else:
                        sentence.append(actions[num])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
                image = cv2.putText(image, ' '.join(sentence), (3,30), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                return redirect("upload-video")
                
                

        cap.release()
        cv2.destroyAllWindows()
        return redirect("upload-video")
    
    return render(request, 'videos/upload.html', context)
