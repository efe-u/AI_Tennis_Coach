import cv2 as cv
import mediapipe as mp
import os
from moviepy.editor import *
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from time import sleep

# This part is directly imported from mediapipe documentation and adjusted accordingly #

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):
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
            solutions.drawing_styles.get_default_pose_landmarks_style())

        # Draw on the extraction
        solutions.drawing_utils.draw_landmarks(
            extraction_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image, extraction_image


FOLDERS = {
    "images": "_annotated",
    "extractions": "_extraction"
}


def deconstruct(FILE_PATH):
    VIDEO_FILE = cv.VideoCapture(FILE_PATH)

    no = 0

    # Folders for video deconstruction
    os.mkdir("images")
    os.mkdir("extractions")

    # Deconstruction into folders
    while True:
        try:
            # Extract images
            ret, frame = VIDEO_FILE.read()
            image = Image.fromarray(frame, 'RGB')
            image = image.save(f"images/{no}.jpeg")
            no += 1
        except:
            break


def configure():
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = python.vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector


def video_annotate(detector):
    no = 0

    for img in tqdm(range(len(os.listdir("images")))):
        # Progress Checker
        image = mp.Image.create_from_file(f"images/{no}.jpeg")
        detection_result = detector.detect(image)

        annotated_image, extraction_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        annotated_image = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)
        os.remove(f"images/{no}.jpeg")
        cv.imwrite(f"images/{no}" + FOLDERS["images"] + ".jpeg", annotated_image)
        cv.imwrite(f"extractions/{no}" + FOLDERS["extractions"] + ".jpeg", extraction_image)
        no += 1


def video_make():
    # Prepare
    for folder in FOLDERS:
        video_name = f'{folder}.avi'

        annotations = []
        for no in tqdm(range(len(os.listdir(folder)))):
            for img in os.listdir(folder):
                if img == f"{no}" + FOLDERS[folder] + ".jpeg":
                    annotations.append(img)

        frame = cv.imread(os.path.join(folder, annotations[0]))
        height, width, layers = frame.shape

        video = cv.VideoWriter(video_name, 0, 1, (width, height))

        for annotation in annotations:
            video.write(cv.imread(os.path.join(folder, annotation)))

        cv.destroyAllWindows()
        video.release()


def video_convert(FILE):
    for folder in FOLDERS:
        video = VideoFileClip(f"{folder}.avi")

        # Adjust video speed
        video = video.set_fps(video.fps * 27)
        video = video.fx(vfx.speedx, 27)

        # Save video
        video.write_videofile(f"results/{FILE}.mp4")
        os.remove(f"{folder}.avi")


# Cleans up folders
def clean():
    for folder in FOLDERS:
        for img in os.listdir(folder):
            os.remove(folder+"/"+img)
        os.rmdir(folder)

    print(folder + " successfully removed!")