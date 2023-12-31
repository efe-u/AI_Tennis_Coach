from PIL import Image
import cv2 as cv
import numpy as np
import mediapipe as mp
from syncronizer import Capture
from tqdm import tqdm


class Video:
    def __init__(self, name, reference, path):
        self.name = name
        self.reference = reference
        self.path = path
        self.Captures = []
        self.All_Normalized_Landmarks = []  # Format: [[[x,y,z],[x,y,z], ... , [x,y,z]], ...] # Dimensions: length x 33 x 3
        self.All_World_Landmarks = []  # Format: [[[x,y,z],[x,y,z], ... , [x,y,z]], ...] # Dimensions: length x 33 x 3
        self.reference_length_candidates = None
        self.position_candidates = None
        self.reference_lengths = []  # [0] -> Normalized reference length, [1] -> World reference length
        self.reference_positions = []  # [0] -> Normalized reference position, [1] -> World reference position
        self.normalization_factors = []  # [0] -> Normalized normalization factor, [1] -> World normalization factor
        self.translation_factor = []  # [0] -> Normalized translation factor, [1] -> World translation factor
        self.mean_errors = []
        self.fps = 30

    def deconstruct(self, shape):
        VIDEO_FILE = cv.VideoCapture(self.path)
        self.fps = int(VIDEO_FILE.get(cv.CAP_PROP_FPS) + 0.5)

        time = 0

        # Deconstruction into folders
        while True:
            # Extract images
            ret, frame = VIDEO_FILE.read()
            if not ret:
                break
            cv.resize(frame, shape, interpolation=cv.INTER_NEAREST)
            self.Captures.append(Capture(time, np.array(frame), self))
            time += 1

    def process(self, detector):
        counter = 0
        for capture in tqdm(self.Captures):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=capture.frame)
            capture.detection_result = detector.detect(image)

            _ = []
            try:
                for landmark in capture.detection_result.pose_landmarks[0]:
                    _.append([landmark.x, landmark.y, landmark.z])
                self.All_Normalized_Landmarks.append(_)

                _ = []
                for landmark in capture.detection_result.pose_world_landmarks[0]:
                    _.append([landmark.x, landmark.y, landmark.z])
                self.All_World_Landmarks.append(_)
                counter += 1

            except:
                self.Captures[counter].detection_result = self.Captures[counter - 1].detection_result
                self.All_Normalized_Landmarks.append(self.All_Normalized_Landmarks[counter - 1])
                self.All_World_Landmarks.append(self.All_World_Landmarks[counter - 1])
                print(f"At {counter} the model had to approximate")
                counter += 1

            if len(self.All_Normalized_Landmarks) != counter:
                print(f"At {counter} normalized landmark is missing")

            if len(self.All_World_Landmarks) != counter:
                print(f"At {counter} world landmark is missing")

    def set_normalization_factors(self, ref):
        if self is not ref:
            normalized_normalization_factor = ref.reference_lengths[0] / self.reference_lengths[0]
            world_normalization_factor = ref.reference_lengths[1] / self.reference_lengths[1]

            self.normalization_factors = [normalized_normalization_factor, world_normalization_factor]
            # Set translation factors
            self.translation_factor.append(self.reference_positions[0]*self.normalization_factors[0] - ref.reference_positions[0])
            self.translation_factor.append(self.reference_positions[1]*self.normalization_factors[0] - ref.reference_positions[1])
        else:
            self.normalization_factors = [1, 1]
            self.translation_factor = [[0, 0, 0], [0, 0, 0]]


def set_average_length(objects):
    for obj in objects:
        obj.reference_lengths = [max(list(obj.reference_length_candidates[0].keys())),
                                 max(list(obj.reference_length_candidates[1].keys()))]

        # reference-point = 27
        obj.reference_positions = [obj.position_candidates[0][0][27],
                                   obj.position_candidates[1][0][27]]


def find_min(objects):
    lengths = []

    for obj in objects:
        lengths.append(obj.reference_lengths[0])

    return objects[lengths.index(min(lengths))]