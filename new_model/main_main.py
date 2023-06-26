import cv2
import mediapipe

from .loading import load_dataset, load_reference_signs ,df
from .mediapipe_use import mediapipe_detection
from .recording import SignRecorder
from .webcam_manager import WebcamManager
from .webcam_manager import videos
from .addon import addons





def prediction(video_name):
    b = True
        # Create dataset of the videos where landmarks have not been extracted yet
    #videos = load_dataset()
    #print(y)
    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    reference_signs = load_reference_signs(videos)
    
    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = SignRecorder(reference_signs)

    # Object that draws keypoints & displays results
    webcam_manager = WebcamManager()
    try:
    # Turn on the webcam
        cap = cv2.VideoCapture(video_name)
        #y = calculate_mean_pixel_value(video_name)
        #x = frame_num(video_name)
        info = addons(video_name)
       
        #result = df.loc[(df["frames"] == x) & (df["mean"] == y), "sign"].values
        # Set up the Mediapipe environment
        with mediapipe.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                
                # Process results
                sign_detected, is_recording = sign_recorder.process_results(results)

                # Update the frame (draw landmarks & display result)
                #webcam_manager.update(frame, results, sign_detected, is_recording)
                
                pressedKey = cv2.waitKey(1) & 0xFF
                #if pressedKey == ord("r"):  # Record pressing r
                sign_recorder.record()
                if pressedKey == ord("q"):# Break pressing q
                    b = False
                    break
                    
            
            cap.release()
            cv2.destroyAllWindows()
            
            
            
    except:
        
        cap.release()
        cv2.destroyAllWindows()

    try:
        if b :
            print(info)
            return info 
    except:
        print('no match signs')
        return 'no match signs'
    
    













