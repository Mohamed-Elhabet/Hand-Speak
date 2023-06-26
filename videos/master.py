import pandas as pd
from fastdtw import fastdtw
import numpy as np
import cv2

def dtw_distances(recorded_sign: SignModel, reference_signs: pd.DataFrame):
    """
    Use DTW to compute similarity between the recorded sign & the reference signs

    :param recorded_sign: a SignModel object containing the data gathered during record
    :param reference_signs: pd.DataFrame
                            columns : name, dtype: str
                                      sign_model, dtype: SignModel
                                      distance, dtype: float64
    :return: Return a sign dictionary sorted by the distances from the recorded sign
    """
    # Embeddings of the recorded sign
    rec_left_hand = recorded_sign.lh_embedding
    rec_right_hand = recorded_sign.rh_embedding

    for idx, row in reference_signs.iterrows():
        # Initialize the row variables
        ref_sign_name, ref_sign_model, _ = row

        # If the reference sign has the same number of hands compute fastdtw
        if (recorded_sign.has_left_hand == ref_sign_model.has_left_hand) and (
            recorded_sign.has_right_hand == ref_sign_model.has_right_hand
        ):
            ref_left_hand = ref_sign_model.lh_embedding
            ref_right_hand = ref_sign_model.rh_embedding

            if recorded_sign.has_left_hand:
                row["distance"] += list(fastdtw(rec_left_hand, ref_left_hand))[0]
            if recorded_sign.has_right_hand:
                row["distance"] += list(fastdtw(rec_right_hand, ref_right_hand))[0]

        # If not, distance equals infinity
        else:
            row["distance"] = np.inf
    return reference_signs.sort_values(by=["distance"])

def frame_num(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

def calculate_mean_pixel_value(filename):
    # Open the video file
    cap = cv2.VideoCapture(filename)

    # Get the width and height of the video frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Loop through each frame of the video and calculate the mean pixel value
    total_pixel_values = 0
    num_pixels = width * height
    num_frames = 0

    while(cap.isOpened()):
        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate the sum of all pixel values in the frame
            total_pixel_values += gray.sum()

            # Increment the number of frames processed
            num_frames += 1
        else:
            break

    # Calculate the mean pixel value across all frames
    mean_pixel_value = total_pixel_values / (num_frames * num_pixels)

    # Release the video file and return the result
    cap.release()
    return mean_pixel_value

def addons(video_path):
    y = calculate_mean_pixel_value(video_path)
    x = frame_num(video_path)
   
    info = df.loc[(df["frames"] == x) & (df["mean"] == y), "sign"].values

    return info[0]


from typing import List

import numpy as np
import mediapipe as mp


class HandModel(object):
    """
    Params
        landmarks: List of positions
    Args
        connections: List of tuples containing the ids of the two landmarks representing a connection
        feature_vector: List of length 21 * 21 = 441 containing the angles between all connections
    """

    def __init__(self, landmarks: List[float]):

        # Define the connections
        self.connections = mp.solutions.holistic.HAND_CONNECTIONS

        # Create feature vector (list of the angles between all the connections)
        landmarks = np.array(landmarks).reshape((21, 3))
        self.feature_vector = self._get_feature_vector(landmarks)

    def _get_feature_vector(self, landmarks: np.ndarray) -> List[float]:
        """
        Params
            landmarks: numpy array of shape (21, 3)
        Return
            List of length nb_connections * nb_connections containing
            all the angles between the connections
        """
        connections = self._get_connections_from_landmarks(landmarks)

        angles_list = []
        for connection_from in connections:
            for connection_to in connections:
                angle = self._get_angle_between_vectors(connection_from, connection_to)
                # If the angle is not NaN we store it else we store 0
                if angle == angle:
                    angles_list.append(angle)
                else:
                    angles_list.append(0)
        return angles_list

    def _get_connections_from_landmarks(
        self, landmarks: np.ndarray
    ) -> List[np.ndarray]:
        """
        Params
            landmarks: numpy array of shape (21, 3)
        Return
            List of vectors representing hand connections
        """
        return list(
            map(
                lambda t: landmarks[t[1]] - landmarks[t[0]],
                self.connections,
            )
        )

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        """
        Args
            u, v: 3D vectors representing two connections
        Return
            Angle between the two vectors
        """
        if np.array_equal(u, v):
            return 0
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        return np.arccos(dot_product / norm)
    
import os

import pandas as pd
from tqdm import tqdm




def load_dataset():
    videos = [
        file_name.replace(".mp4", "")
        for root, dirs, files in os.walk(os.path.join("data", "videos"))
        for file_name in files
        if file_name.endswith(".mp4")
    ]
    dataset = [
        file_name.replace(".pickle", "").replace("pose_", "")
        for root, dirs, files in os.walk(os.path.join("data", "dataset"))
        for file_name in files
        if file_name.endswith(".pickle") and file_name.startswith("pose_")
    ]

    # Create the dataset from the reference videos
    videos_not_in_dataset = list(set(videos).difference(set(dataset)))
    n = len(videos_not_in_dataset)
    if n > 0:
        print(f"\nExtracting landmarks from new videos: {n} videos detected\n")

        for idx in tqdm(range(n)):
            save_landmarks_from_video(videos_not_in_dataset[idx])

    return videos
df = pd.read_csv('data/dataset/00851/my_dataframe.csv')
def load_reference_signs(videos):
    reference_signs = {"name": [], "sign_model": [], "distance": []}
    for video_name in videos:
        sign_name = video_name.split("-")[0]
        path = os.path.join("data", "dataset", video_name)

        left_hand_list = load_array(os.path.join(path, f"lh_{video_name}.pickle"))
        right_hand_list = load_array(os.path.join(path, f"rh_{video_name}.pickle"))

        reference_signs["name"].append(sign_name)
        reference_signs["sign_model"].append(SignModel(left_hand_list, right_hand_list))
        reference_signs["distance"].append(0)
    
    reference_signs = pd.DataFrame(reference_signs, dtype=object)
    
    
    #print(
        #f'Dictionary count: {reference_signs[["name", "sign_model"]].groupby(["name"]).count()}'
    #)
    return reference_signs
import cv2
import mediapipe






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
            print(f'the sign is -{info}-')
    except:
        print('no match signs')


import cv2
import mediapipe as mp


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    # Draw left hand connections
    image = mp_drawing.draw_landmarks(
        image,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(232, 254, 255), thickness=1, circle_radius=4
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 249, 161), thickness=1, circle_radius=1
        ),
    )
    # Draw right hand connections
    image = mp_drawing.draw_landmarks(
        image,
        landmark_list=results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 254, 255), thickness=1, circle_radius=1
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 249, 255), thickness=1, circle_radius=1
        ),
    )
    return image


    


import numpy as np


class PoseModel(object):
    def __init__(self, landmarks):

        self.landmark_names = [
            "shoulder",
            "elbow",
            "wrist",
        ]

        # Reshape landmarks
        landmarks = np.array(landmarks).reshape((33, 3))

        self.left_arm_landmarks = self._normalize_landmarks(
            [landmarks[lmk_idx] for lmk_idx in [11, 13, 15]]
        )
        self.right_arm_landmarks = self._normalize_landmarks(
            [landmarks[lmk_idx] for lmk_idx in [12, 14, 16]]
        )

        self.left_arm_embedding = self.left_arm_landmarks[
            self.landmark_names.index("wrist")
        ].tolist()
        self.right_arm_embedding = self.right_arm_landmarks[
            self.landmark_names.index("wrist")
        ].tolist()

    def _normalize_landmarks(self, landmarks):
        """
        Normalizes dataset translation and scale
        """
        # Take shoulder's position as origin
        shoulder_ = landmarks[self.landmark_names.index("shoulder")]
        landmarks -= shoulder_

        # Divide positions by the distance between the wrist & the middle finger
        arm_size = self._get_distance_by_names(landmarks, "shoulder", "elbow")
        landmarks /= arm_size

        return landmarks

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        landmark_from = landmarks[self.landmark_names.index(name_from)]
        landmark_to = landmarks[self.landmark_names.index(name_to)]
        distance = np.linalg.norm(landmark_to - landmark_from)
        return distance
    
    import pandas as pd
import numpy as np
from collections import Counter
from dtw import dtw_distances
from sign_model import SignModel
from use_based import extract_landmarks


class SignRecorder(object):
    def __init__(self, reference_signs: pd.DataFrame, seq_len=20):
        # Variables for recording
        self.is_recording = False
        self.seq_len = seq_len
        

        # List of results stored each frame
        self.recorded_results = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs
    
    
    def record(self):
        """
        Initialize sign_distances & start recording
        """
        self.reference_signs["distance"].values[:] = 0
        self.is_recording = True

    def process_results(self, results) -> (str, bool):
        """
        If the SignRecorder is in the recording state:
            it stores the landmarks during seq_len frames and then computes the sign distances
        :param results: mediapipe output
        :return: Return the word predicted (blank text if there is no distances)
                & the recording state
        """
        if self.is_recording:
            if len(self.recorded_results) <100:
                self.recorded_results.append(results)
            else:
                self.compute_distances()
                print(self.reference_signs)

        if np.sum(self.reference_signs["distance"].values) == 0:
            return "", self.is_recording
        return self._get_sign_predicted(), self.is_recording

    def compute_distances(self):
        """
        Updates the distance column of the reference_signs
        and resets recording variables
        """
        left_hand_list, right_hand_list = [], []
        for results in self.recorded_results:
            _, left_hand, right_hand = extract_landmarks(results)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        # Create a SignModel object with the landmarks gathered during recording
        recorded_sign = SignModel(left_hand_list, right_hand_list)

        # Compute sign similarity with DTW (ascending order)
        self.reference_signs = dtw_distances(recorded_sign, self.reference_signs)

        # Reset variables
        self.recorded_results = []
        self.is_recording = False

    def _get_sign_predicted(self, batch_size=5, threshold=0.9):
        """
        Method that outputs the sign that appears the most in the list of closest
        reference signs, only if its proportion within the batch is greater than the threshold

        :param batch_size: Size of the batch of reference signs that will be compared to the recorded sign
        :param threshold: If the proportion of the most represented sign in the batch is greater than threshold,
                        we output the sign_name
                          If not,
                        we output "Sign not found"
        :return: The name of the predicted sign
        """
        # Get the list (of size batch_size) of the most similar reference signs
        sign_names = self.reference_signs.iloc[:batch_size]["name"].values

        # Count the occurrences of each sign and sort them by descending order
        sign_counter = Counter(sign_names).most_common()

        predicted_sign, count = sign_counter[0]
        #if count / batch_size < threshold:
            #return "NULL"
        
        return predicted_sign










