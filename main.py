from pose_extraction import *
from syncronizer import *
from process import *
from time_sync import *

FILES = [
    Video("alcaraz_serve", "media/alcaraz_serve-synced.mp4"),
    Video("kid_serve", "media/test_4.mp4")
]

if __name__ == '__main__':
    FILES[0].deconstruct()
    FILES[1].deconstruct()

    detector = configure()

    FILES[0].process(detector)
    FILES[1].process(detector)

    ref = get_shorter(FILES)
    FILES[0].reference_length_candidates, FILES[0].position_candidates = pre_normalize(FILES[0], ref)
    FILES[1].reference_length_candidates, FILES[1].position_candidates = pre_normalize(FILES[1], ref)

    # TODO: TIME SYNCRONIZER
    frames1, frames2 = pre_sync(FILES, ref)
    mean_error(frames1, frames2)


    set_average_length(FILES)
    ref = find_min(FILES)

    video_annotate(detector, FILES[0], ref, (255,0,0))
    video_make(FILES[0])
    video_convert(FILES[0])
    clean()

    video_annotate(detector, FILES[1], ref, (0,0,255))
    video_make(FILES[1])
    video_convert(FILES[1])
    clean()

    blend(FILES)

    print("So far, so good")