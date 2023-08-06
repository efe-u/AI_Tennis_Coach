from mediapipe.tasks.python.vision import pose_landmarker
from mediapipe.tasks.python.components.containers import landmark
import numpy as np
import math


class Capture():
    def __init__(self, time, frame, video):
        self.video = video # Super video
        self.time = time # Time frame in video
        self.frame = frame # Image array
        self.detection_result = None
        self.NormalizedLandmarks = []
        self.Landmarks = []

    # Reference length is taken as the distance between the shoulders

    def set_reference_lengths(self):
        u = self.video.All_Normalized_Landmarks[self.time] # All normalized landmarks in this frame

        normalize_reference_length = [u[12][0] - u[11][0], # x
                                      u[12][0] - u[11][0], # y
                                      u[12][0] - u[11][0]] # z

        u = self.video.All_World_Landmarks[self.time] # All world landmarks in this frame

        world_reference_length = [u[12][0] - u[11][0], # x
                                  u[12][0] - u[11][0], # y
                                  u[12][0] - u[11][0]] # z

        self.video.reference_lengths = [normalize_reference_length, world_reference_length]

        return [math.sqrt(self.video.reference_lengths[0][0] ** 2 +
                          self.video.reference_lengths[0][1] ** 2 +
                          self.video.reference_lengths[0][2] ** 2),
                math.sqrt(self.video.reference_lengths[1][0] ** 2 +
                          self.video.reference_lengths[1][1] ** 2 +
                          self.video.reference_lengths[1][2] ** 2)]

    def normalize(self, ref):
        if self.time == 1:
            pass

        _ = [[], []]

        for mark in np.array(self.video.All_Normalized_Landmarks[self.time]): # All normalized landmarks in this frame
            mark *= self.video.normalization_factors[0]

            # Translate
            mark -= np.array(self.video.translation_factor[0])

            _[0].append(mark)

        for mark in np.array(self.video.All_World_Landmarks[self.time]): # All normalized landmarks in this frame
            mark *= self.video.normalization_factors[1]

            # Translate
            mark -= np.array(self.video.translation_factor[1])

            _[1].append(mark)

        self.video.All_Normalized_Landmarks[self.time] = _[0]


    def get_normalized_PoseLandmarkerResult(self, ref) -> pose_landmarker.PoseLandmarkerResult:
        self.normalize(ref)

        self.NormalizedLandmarks = []

        for mark in self.video.All_Normalized_Landmarks[self.time]:
            self.NormalizedLandmarks.append(landmark.NormalizedLandmark(x = mark[0],
                                                                        y = mark[1],
                                                                        z = mark[2],
                                                                        visibility = self.detection_result.pose_landmarks[0][len(self.NormalizedLandmarks)].visibility,
                                                                        presence = self.detection_result.pose_landmarks[0][len(self.NormalizedLandmarks)].presence))
        self.Landmarks = []

        for mark in self.video.All_World_Landmarks[self.time]:
            self.Landmarks.append(landmark.Landmark(x = mark[0],
                                                    y = mark[1],
                                                    z = mark[2],
                                                    visibility = self.detection_result.pose_world_landmarks[0][len(self.Landmarks)].visibility,
                                                    presence = self.detection_result.pose_world_landmarks[0][len(self.Landmarks)].presence))

        pose_landmarker_result = pose_landmarker.PoseLandmarkerResult([self.NormalizedLandmarks], [self.Landmarks])
        return pose_landmarker_result