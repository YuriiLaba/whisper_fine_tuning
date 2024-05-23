from src.extract_youtube_playlist import download_playlist
from src.utils import create_directory
from configs import *

if __name__ == "__main__":
    create_directory(path_to_videos)
    create_directory(path_to_captions)

    download_playlist(playlist_link, path_to_videos, path_to_captions, path_to_meta_info, file_prefix)