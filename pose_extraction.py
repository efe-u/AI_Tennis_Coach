from syncronizer import *
import cv2 as cv
import mediapipe as mp
import os
from moviepy.editor import *
from error_calculator import calculate_error, visualize
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from mediapipe import solutions
import mediapipe.tasks.python.components.containers.landmark
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result, color, factors, capture, VIDEO):
    landmarks = capture.NormalizedLandmarks
    annotated_image = np.copy(rgb_image)
    extraction_image = np.zeros((annotated_image.shape))

    # Segmentation Mask Processing
    # Normalization
    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    segmentation_image = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    u = segmentation_image.shape
    segmentation_image = np.concatenate(
        (segmentation_image, np.zeros((int((u[0] / factors[0] - u[0])), segmentation_image.shape[1], 3))), axis=0)
    segmentation_image = np.concatenate(
        (segmentation_image, np.zeros((segmentation_image.shape[0], int((u[1] / factors[0] - u[1])), 3))), axis=1)

    segmentation_image = cv.resize(segmentation_image, u[:2], interpolation=cv.INTER_NEAREST)

    # Translation
    dx = -int(u[0] * factors[1][0])
    if dx < 0:
        segmentation_image = np.delete(segmentation_image, slice(0, abs(dx)), 1)
        segmentation_image = np.concatenate((segmentation_image, np.zeros((u[1], abs(dx), 3))), axis=1)
    else:
        segmentation_image = np.delete(segmentation_image, slice(u[0] - dx, u[0]+1), 1)
        segmentation_image = np.concatenate((np.zeros((u[1], abs(dx), 3)), segmentation_image), axis=1)

    dy = int(u[1] * factors[1][1])
    if dy > 0:
        segmentation_image = np.delete(segmentation_image, slice(0, abs(dy)), 0)
        segmentation_image = np.concatenate((segmentation_image, np.zeros((abs(dy), u[0], 3))), axis=0)
    else:
        segmentation_image = np.delete(segmentation_image, slice(u[1] + dy, u[1]+1), 0)
        segmentation_image = np.concatenate((np.zeros((abs(dy), u[0], 3)), segmentation_image), axis=0)


    # Loop through the detected poses to visualize.
    # Draw the pose landmarks.
    trace = landmark_pb2.NormalizedLandmarkList()
    trace.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=capture.video.Captures[t].NormalizedLandmarks[16].x, y=capture.video.Captures[t].NormalizedLandmarks[16].y, z=capture.video.Captures[t].NormalizedLandmarks[16].z) for t in range(capture.time)
    ])

    pose_landmarks_image = landmark_pb2.NormalizedLandmarkList()

    pose_landmarks_image.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=mark[0], y=mark[1], z=mark[2]) for mark in capture.video.All_Normalized_Landmarks[capture.time]
    ])

    pose_landmarks_ext = landmark_pb2.NormalizedLandmarkList()

    pose_landmarks_ext.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=mark.x, y=mark.y, z=mark.z) for mark in landmarks
    ])

    try:
        mean_error, to_mark = calculate_error(capture, VIDEO)
    except:
        mean_error = None
        to_mark = False

    if not not to_mark:
        _ = []
        for mark in to_mark:
            _.append(landmarks[mark[0]])
            _.append(landmarks[mark[1]])

        corrections = landmark_pb2.NormalizedLandmarkList()
        corrections.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=mark.x, y=mark.y, z=mark.z) for mark in _
        ])

    # Draw on the image
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_image,
        solutions.pose.POSE_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(color, 3, 3))

    # Draw on the extraction
    solutions.drawing_utils.draw_landmarks(
        extraction_image,
        pose_landmarks_ext,
        solutions.pose.POSE_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(color, 3, 3))
    solutions.drawing_utils.draw_landmarks(
        image = extraction_image,
        landmark_list = trace,
        landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec((0,255,0), 1, 1))

    # Draw on segmentation
    solutions.drawing_utils.draw_landmarks(
        segmentation_image,
        pose_landmarks_ext,
        solutions.pose.POSE_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(color, 3, 3))
    solutions.drawing_utils.draw_landmarks(
        image = segmentation_image,
        landmark_list = trace,
        landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec((0,255,0), 1, 1))

    if not not to_mark:
        for mark in to_mark:
            x = to_mark[mark]
            solutions.drawing_utils.draw_landmarks(
                image=extraction_image,
                landmark_list=pose_landmarks_ext,
                connections=[mark],
                connection_drawing_spec=(mp.solutions.drawing_utils.DrawingSpec((0, 255-x, x), 2, 2)),
                landmark_drawing_spec=None)
            solutions.drawing_utils.draw_landmarks(
                image=segmentation_image,
                landmark_list=pose_landmarks_ext,
                connections=[mark],
                connection_drawing_spec=(mp.solutions.drawing_utils.DrawingSpec((0, 255-x, x), 2, 2)),
                landmark_drawing_spec=None)

        VIDEO.mean_errors.append(mean_error)
        extraction_image, segmentation_image = visualize(capture, VIDEO.mean_errors, extraction_image, segmentation_image)
    else:
        extraction_image = np.concatenate((extraction_image, np.zeros((int(extraction_image.shape[0] / 5), extraction_image.shape[1], 3))), axis=0)
        segmentation_image = np.concatenate((segmentation_image, np.zeros((int(segmentation_image.shape[0] / 5), segmentation_image.shape[1], 3))), axis=0)

    return annotated_image, extraction_image, segmentation_image


def configure():
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = python.vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_presence_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector


def pre_normalize(VIDEO, ref):
    stable_references = [{}, {}]
    stable_coordinates = [[], []]

    step = int(len(ref.Captures)*0.1)

    # for every approx. 10% period of the shorter video
    for aggregate in range(0, len(VIDEO.Captures) - (len(VIDEO.Captures) % step), step):
        average_references = [[], []]
        average_coordinates = np.zeros((2,33,3))

        # for every image in the approx. 10% of the shorter video
        for capture in range(aggregate, aggregate + step):

            u = VIDEO.Captures[capture] # Reference to the frame at the current time

            # Normalization
            average_references[0].append(u.set_reference_lengths()[0])
            average_references[1].append(u.set_reference_lengths()[1])

            # Translation
            for i in range(len(VIDEO.All_Normalized_Landmarks[u.time])):
                average_coordinates[0][i] += np.array(VIDEO.All_Normalized_Landmarks[u.time][i]) / step

            for i in range(len(VIDEO.All_World_Landmarks[u.time])):
                average_coordinates[1][i] += np.array(VIDEO.All_World_Landmarks[u.time][i]) / step


        stable_references[0][np.mean(np.array(average_references[0]))] = np.std(np.array(average_references[0]))
        stable_references[1][np.mean(np.array(average_references[1]))] = np.std(np.array(average_references[1]))

        stable_coordinates[0].append(average_coordinates[0])
        stable_coordinates[1].append(average_coordinates[1])
    return stable_references, stable_coordinates


def video_annotate(VIDEO, ref, color, shape):

    VIDEO.set_normalization_factors(ref)

    for capture in tqdm(VIDEO.Captures):
        detection_result = capture.get_normalized_PoseLandmarkerResult()

        annotated_image, extraction_image, segmentation_image = draw_landmarks_on_image(capture.frame, detection_result, color, [VIDEO.normalization_factors[0], VIDEO.translation_factor[0]], capture, ref)
        annotated_image = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)

        cv.imwrite(f"images/{capture.time}.jpeg", cv.resize(annotated_image, shape, interpolation=cv.INTER_AREA))
        cv.imwrite(f"extractions/{capture.time}.jpeg", cv.resize(extraction_image, shape, interpolation=cv.INTER_AREA))
        cv.imwrite(f"segmentations/{capture.time}.jpeg", cv.resize(segmentation_image, shape, interpolation=cv.INTER_AREA))


def video_make(VIDEO):
    # Prepare
    for folder in ["images", "extractions", "segmentations"]:
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


def video_convert(VIDEO, ref):
    for folder in ["images", "extractions", "segmentations"]:
        video = VideoFileClip(VIDEO.name + f"_{folder}.avi")

        # Adjust video speed
        video = video.set_fps(len(VIDEO.Captures)/10)
        video = video.fx(vfx.speedx, (len(VIDEO.Captures)/len(ref.Captures)) * len(ref.Captures) * 0.1)

        # Save video
        video.write_videofile(f"results/{VIDEO.name}({folder}).mp4")
        os.remove(VIDEO.name + f"_{folder}.avi")

1
def blend(VIDEOS, ref):
    clip1 = VideoFileClip("results/" + VIDEOS[0].name + "(extractions).mp4")
    clip2 = VideoFileClip("results/" + VIDEOS[1].name + "(extractions).mp4")
    clip3 = VideoFileClip("results/" + VIDEOS[0].name + "(images).mp4")
    clip4 = VideoFileClip("results/" + VIDEOS[1].name + "(images).mp4")
    clip5 = VideoFileClip("results/" + VIDEOS[0].name + "(segmentations).mp4")
    clip6 = VideoFileClip("results/" + VIDEOS[1].name + "(segmentations).mp4")


    final_clip = clips_array([[clip1, clip2]])
    final_clip.write_videofile("results/blend1.1.mp4")

    final_clip = clips_array([[clip5, clip6]])
    final_clip.write_videofile("results/blend1.2.mp4")

    clip2 = clip2.set_opacity(0.5)

    final_clip = CompositeVideoClip([clip1, clip2])
    final_clip.write_videofile("results/blend2.mp4")

    if VIDEOS[1].reference is True:
        clip5 = clip5.set_opacity(0.5)

        final_clip = CompositeVideoClip([clip4, clip5])
        final_clip.write_videofile("results/blend3.1.mp4")

    if VIDEOS[0].reference is True:
        clip6 = clip6.set_opacity(0.5)

        final_clip = CompositeVideoClip([clip3, clip6])
        final_clip.write_videofile("results/blend3.2.mp4")


# Cleans up folders
def clean():
    for folder in ["images", "extractions", "segmentations"]:
        for img in os.listdir(folder):
            os.remove(folder + "/" + img)
