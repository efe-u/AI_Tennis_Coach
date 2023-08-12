from pose_extraction import *
from syncronizer import *
from process import *
from time_sync import *

if __name__ == '__main__':
    FILES = [
        Video("1", "media/djokovic_forehand-synced.mp4"),
        Video("2", "media/ksenia_forehand.mp4")
    ]

    shape = (850, 380)

    FILES[0].deconstruct(shape)
    FILES[1].deconstruct(shape)

    detector = configure()

    FILES[0].process(detector)
    FILES[1].process(detector)

    ref = get_shorter(FILES)
    FILES[0].reference_length_candidates, FILES[0].position_candidates = pre_normalize(FILES[0], ref)
    FILES[1].reference_length_candidates, FILES[1].position_candidates = pre_normalize(FILES[1], ref)

    # TODO: TIME SYNCRONIZER
    frames1, frames2 = pre_sync(FILES, ref)
    modify_results(FILES, ref, frames1, frames2)

    set_average_length(FILES)

    video_annotate(detector, FILES[0], find_min(FILES), (255,0,0), shape)
    video_make(FILES[0])
    video_convert(FILES[0], ref)
    clean()

    video_annotate(detector, FILES[1], find_min(FILES), (0,0,255), shape)
    video_make(FILES[1])
    video_convert(FILES[1], ref)
    clean()

    blend(FILES, ref)

    print("So far, so good")