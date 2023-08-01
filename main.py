from pose_extraction import *

FILES = {
    "alcaraz_serve":"media/alcaraz_serve-synced.mp4",
    "kid_serve":"media/kid_serve-synced.mp4"
}

if __name__ == '__main__':
    for FILE in FILES:
        FILE_PATH = FILES[FILE]
        deconstruct(FILE_PATH)
        detector = configure()
        video_annotate(detector)
        video_make()
        video_convert(FILE)
        clean()