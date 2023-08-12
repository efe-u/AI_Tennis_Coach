import cv2 as cv
from process import Video
import numpy as np
import math

def get_shorter(FILES):
    min = FILES[0]

    for video in FILES:
        if len(video.Captures) < len(min.Captures):
            min = video

    return min
def set_factor(obj):
    return max(list(obj.reference_length_candidates[0].keys()))

def magnitude(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

def get_translation_point(objects):
    indices = []
    for obj in objects:
        _ = np.empty((33,len(obj.position_candidates[0])), dtype=list)

        for i in range(len(_)):
            for j in range(len(obj.position_candidates[0])):
                _[i][j] = magnitude(obj.position_candidates[0][j][i])

        min = [1]*10
        ind = [0]*10
        for i in range(len(_)):
            if np.std(_[i]) < max(min):
                min[min.index(max(min))] = np.std(_[i])
                ind[min.index(max(min))] = i

        temp = np.array(min).argsort()
        ind = np.array(ind)[temp]

        indices.append(ind)
    return np.intersect1d(np.array(indices[0]), np.array(indices[1]))[0]

def pre_sync(FILES, shorter):
    # Initial positions
    frames2 = np.copy(shorter.position_candidates[0])
    for file in FILES:
        if file is not shorter:
            frames1 = np.copy(file.position_candidates[0])

    # Normalization according to max characteristic length
    factor1 = set_factor(FILES[0])

    for i in range(len(frames1)):
        for j in range(len(frames1[i])):
            frames1[i][j] /= factor1

    factor2 = set_factor(FILES[1])

    for i in range(len(frames2)):
        for j in range(len(frames2[i])):
            frames2[i][j] /= factor2

    # Translation according to least moving part
    ref = get_translation_point(FILES)

    for i in range(len(frames2)):
        translation = frames1[i][ref] - frames2[i][ref]
        for j in range(len(frames2[i])):
            frames2[i][j] += translation

    return frames1, frames2

def mean_error(frames1, frames2):
    # Adjusted weights for movements
    serve = [10, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 5, 10, 8, 15, 10, 15, 1, 1, 1, 1, 0, 0, 5, 10, 2, 15, 10, 15, 10, 10, 0, 10]
    forehand = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 15, 10, 20, 5, 15, 1, 1, 1, 1, 0, 0, 5, 15, 5, 5, 3, 5, 0, 5, 0, 0]

    # Set weights
    weights = forehand

    # frame1 to frame2 comparison
    errors = np.zeros((2, len(frames2), len(frames2[0])), dtype=list)

    for i in range(len(frames2)):
        for j in range(len(frames2[i])):
            errors[0][i][j] = magnitude(frames1[0][j] - frames2[i][j]) * weights[j]
            errors[1][i][j] = magnitude(frames1[-1][j] - frames2[i][j]) * weights[j]

    s1 = []
    f1 = []
    for i in range(len(errors[0])):
        s1.append(np.mean(errors[0][i]))
        f1.append(np.mean(errors[1][i]))

    # frame2 to frame1 comparison
    errors = np.zeros((2, len(frames1), len(frames1[0])), dtype=list)

    for i in range(len(frames1)):
        for j in range(len(frames1[i])):
            errors[0][i][j] = magnitude(frames2[0][j] - frames1[i][j]) * weights[j]
            errors[1][i][j] = magnitude(frames2[-1][j] - frames1[i][j]) * weights[j]

    s2 = []
    f2 = []
    for i in range(len(errors[0])):
        s2.append(np.mean(errors[0][i]))
        f2.append(np.mean(errors[1][i]))

    s1 = s1.index(min(s1))
    s2 = s2.index(min(s2))
    f1 = f1.index(min(f1))
    f2 = f2.index(min(f2))

    return s1,s2,f1,f2

def modify_results(FILES, ref, frames1, frames2):
    s1, s2, f1, f2 = mean_error(frames1, frames2)

    step = len(ref.Captures)*0.1

    for file in FILES:
        if file is not ref:
            file.reference_length_candidates[0] = dict(zip(list(file.reference_length_candidates[0].keys())[s2:(f2+1)],
                                                            list(file.reference_length_candidates[0].values())[s2:(f2+1)]))
            file.reference_length_candidates[1] = dict(zip(list(file.reference_length_candidates[1].keys())[s2:(f2+1)],
                                                           list(file.reference_length_candidates[1].values())[s2:(f2+1)]))
            file.position_candidates[0] = file.position_candidates[0][s2:(f2+1)]
            file.position_candidates[1] = file.position_candidates[1][s2:(f2 + 1)]

            file.All_Normalized_Landmarks = file.All_Normalized_Landmarks[int(s2*len(ref.Captures)/10):int((f2+1)*len(ref.Captures)/10)]
            file.Captures = file.Captures[int(s2*step):int((f2+1)*step)]

            for i in range(len(file.Captures)):
                file.Captures[i].time = i

        else:
            file.reference_length_candidates[0] = dict(zip(list(file.reference_length_candidates[0].keys())[s1:(f1+1)],
                                                           list(file.reference_length_candidates[0].values())[s1:(f1+1)]))
            file.reference_length_candidates[1] = dict(zip(list(file.reference_length_candidates[1].keys())[s1:(f1+1)],
                                                           list(file.reference_length_candidates[1].values())[s1:(f1+1)]))
            file.position_candidates[0] = file.position_candidates[0][s1:(f1+1)]
            file.position_candidates[1] = file.position_candidates[1][s1:(f1 + 1)]

            file.All_Normalized_Landmarks = file.All_Normalized_Landmarks[int(s1*len(ref.Captures)/10):int((f1+1)*len(ref.Captures)/10)]
            file.Captures = file.Captures[int(s1*step):int((f1+1)*step)]

            for i in range(len(file.Captures)):
                file.Captures[i].time = i

    print("So far, so good")
