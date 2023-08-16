from time_sync import magnitude
import numpy as np
from mediapipe import solutions
import matplotlib.pyplot as plt

def calculate_error(capture, VIDEO):
    if not VIDEO.reference and capture.video.reference:
        return

    factor = len(VIDEO.Captures)/len(capture.video.Captures)

    u1 = capture.video.All_Normalized_Landmarks[capture.time]
    u2 = VIDEO.All_Normalized_Landmarks[int(capture.time * factor)]

    errors = np.array(u1) - np.array(u2)

    _ = []
    for i in range(len(errors)):
        _.append(magnitude(errors[i]))
    errors = _

    mean_error = np.mean(errors)

    connection_errors = []
    for connection in solutions.pose.POSE_CONNECTIONS:
        connection_errors.append((errors[connection[0]] + errors[connection[1]])*0.5)


    scaled_connection_errors = []
    for error in connection_errors:
        scaled_connection_errors.append((error - min(connection_errors)) / (max(connection_errors) - min(connection_errors)))

    _ = []
    __ = []
    for i in range(len(scaled_connection_errors)):
        _.append(list(solutions.pose.POSE_CONNECTIONS)[scaled_connection_errors.index(max(scaled_connection_errors))])
        __.append(max(scaled_connection_errors)*255)
        scaled_connection_errors[scaled_connection_errors.index(max(scaled_connection_errors))] = -1

    to_mark = dict(zip(_, __))

    return mean_error, to_mark


def visualize(capture, mean_errors, extraction_image, segmentation_image):
    images = [extraction_image, segmentation_image]

    for img in range(len(images)):
        fig = plt.figure(figsize=(images[img].shape[1] / 100, images[img].shape[0] / 500), facecolor="grey")
        fig.add_subplot(121)
        plt.xlim([0, len(capture.video.Captures)])
        plt.ylim([0, 1])
        plt.plot(mean_errors)

        fig.add_subplot(122)
        plt.ylim([0, 1])
        plt.bar(1, mean_errors[-1])

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        images[img] = np.concatenate((images[img], image_from_plot), axis=0)

    return images[0], images[1]