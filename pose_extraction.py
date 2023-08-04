from syncronizer import *
import cv2 as cv
import mediapipe as mp
import os
from moviepy.editor import *
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

# This part is directly imported from mediapipe documentation and adjusted accordingly #

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result, color):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    extraction_image = np.zeros(annotated_image.shape)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])

        # arguments(image, landmark_positions, connect_landmarks, connection_style)

        # Draw on the image
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color, 3, 3))

        # Draw on the extraction
        solutions.drawing_utils.draw_landmarks(
            extraction_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color, 3, 3))
    return annotated_image, extraction_image


def configure():
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = python.vision.PoseLandmarkerOptions(
        base_options=base_options)
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector


def pre_normalize(VIDEO):
    stable_references = [{}, {}]
    stable_coordinates = [[], []]

    step = int(len(VIDEO.Captures)*0.1)

    # for every approx. 10% of all images
    for aggregate in tqdm(range(0, len(VIDEO.Captures) - (len(VIDEO.Captures) % 10), step)):
        average_references = [[], []]
        average_coordinates = np.array([[0.0,0.0,0.0], [0.0,0.0,0.0]])

        # for every image in the approx. 10%
        for capture in range(aggregate, aggregate + int(len(VIDEO.Captures)*0.1)):

            u = VIDEO.Captures[capture] # Reference to the frame at the current time

            # Normalization
            average_references[0].append(u.set_reference_lengths()[0])
            average_references[1].append(u.set_reference_lengths()[1])

            # Translation (reference-point = 27)
            v = VIDEO.All_Normalized_Landmarks[u.time] # Reference to the normalized landmarks at the current time
            average_coordinates[0] += np.array(v[27]) / step

            v = VIDEO.All_World_Landmarks[u.time]
            average_coordinates[1] += np.array(v[27]) / step


        stable_references[0][np.mean(np.array(average_references[0]))] = np.std(np.array(average_references[0]))
        stable_references[1][np.mean(np.array(average_references[1]))] = np.std(np.array(average_references[1]))

        stable_coordinates[0].append(average_coordinates[0])
        stable_coordinates[1].append(average_coordinates[1])
    return stable_references, stable_coordinates


def video_annotate(detector, VIDEO, ref, color):

    VIDEO.set_normalization_factors(ref)

    for capture in tqdm(VIDEO.Captures):
        detection_result = capture.get_normalized_PoseLandmarkerResult()

        annotated_image, extraction_image = draw_landmarks_on_image(capture.frame, detection_result, color)
        annotated_image = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)

        cv.imwrite(f"images/{capture.time}.jpeg", annotated_image)
        cv.imwrite(f"extractions/{capture.time}.jpeg", extraction_image)


def video_make(VIDEO):
    # Prepare
    for folder in ["images", "extractions"]:
        video_name = VIDEO.name + f'_{folder}.avi'

        annotations = []
        for no in tqdm(range(len(os.listdir(folder)))):
            for img in os.listdir(folder):
                if img == f"{no}.jpeg":
                    annotations.append(img)

        frame = cv.imread(os.path.join(folder, annotations[0]))
        height, width, layers = frame.shape

        video = cv.VideoWriter(video_name, 0, 1, (width, height))

        for annotation in annotations:
            video.write(cv.imread(os.path.join(folder, annotation)))

        cv.destroyAllWindows()
        video.release()


def video_convert(VIDEO):
    for folder in ["images", "extractions"]:
        video = VideoFileClip(VIDEO.name + f"_{folder}.avi")

        # Adjust video speed
        video = video.set_fps(video.fps * VIDEO.fps)
        video = video.fx(vfx.speedx, VIDEO.fps)

        # Save video
        video.write_videofile(f"results/{VIDEO.name}({folder}).mp4")
        os.remove(VIDEO.name + f"_{folder}.avi")


def blend(VIDEOS):
    clip1 = VideoFileClip("results/" + VIDEOS[0].name + "(extractions).mp4")
    clip2 = VideoFileClip("results/" + VIDEOS[1].name + "(extractions).mp4")

    final_clip = clips_array([[clip1, clip2]])
    final_clip.write_videofile("results/blend1.mp4")

    clip2 = clip2.set_opacity(0.5)

    final_clip = CompositeVideoClip([clip1, clip2])
    final_clip.write_videofile("results/blend2.mp4")


# Cleans up folders
def clean():
    for folder in ["images", "extractions"]:
        for img in os.listdir(folder):
            os.remove(folder + "/" + img)
