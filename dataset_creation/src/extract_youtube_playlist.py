from pytube import Playlist
from pytube.exceptions import AgeRestrictedError
import xmltodict
import json
import srt
from tqdm import tqdm
from datetime import timedelta
import os


# I think that it may make sense to rewrite this code into YouTube DL. I see a couple of bottlenecks
# with this implementation. We need to download video to get audio. Why can't we just get audio from the beginning.
# We need to process audio (convert to wav, etc) using ffmpeg. As far as I understand we can do it on the spot with
# This code has to be rewritten into class


def srt_from_xml(xml):
    xml_dict = xmltodict.parse(xml)
    rows = []
    for i, row in enumerate(xml_dict["timedtext"]["body"]["p"]):
        start = timedelta(milliseconds=int(row["@t"]))
        end = timedelta(milliseconds=int(row["@t"])+int(row["@d"]))
        rows.append(srt.Subtitle(index=i, start=start, end=end, content=row["#text"].replace("\n", " ")))
    return srt.compose(rows)


def get_video_stream(video):
    try:
        return video.streams.filter(progressive=True, file_extension='mp4').first()
    except AgeRestrictedError:
        # TODO: add logger
        print(AgeRestrictedError)
        return None


def get_captions(video):
    try:
        captions = video.captions['uk']
        captions_srt = srt_from_xml(captions.xml_captions)
        return captions_srt

    except KeyError:
        # TODO: add logger
        print("No captions")
        return None


def save_video(video_stream, path_to_videos, filename_to_save):
    video_filename = f"{filename_to_save}.mp4"

    video_stream.download(output_path=path_to_videos, filename=video_filename)


def save_captions(captions, path_to_captions, filename_to_save):
    caption_filename = f"{filename_to_save}.srt"

    with open(f"{path_to_captions}/{caption_filename}", "w") as file:
        file.write(captions)
    print(f"{path_to_captions}/{caption_filename}")


def download_playlist(playlist_link, path_to_videos, path_to_captions, path_to_meta_info, file_prefix):

    playlist = Playlist(playlist_link)
    meta_info = []

    # TODO: remove this, add to meta info
    total_video_count = 0
    total_video_parsed = 0

    print(f'Downloading videos by: {playlist.title}')
    for idx, video in enumerate(tqdm(playlist.videos, total=len(playlist.videos), desc="Processing videos")):
        print(video.title)
        total_video_count += 1

        filename_to_save = f"{file_prefix}_{idx}"

        meta_info.append({
            "title": str(video.title),
            "captions_path": f"{path_to_captions}/{filename_to_save}.srt",
            "video": f"{path_to_videos}/{filename_to_save}.mp4",
        })

        if os.path.exists(f"{path_to_videos}/{filename_to_save}.mp4"):
            continue

        video_stream = get_video_stream(video)  # I don't understand why I can't have captions without extracting stream
        if video_stream is None:
            continue

        captions = get_captions(video)
        if captions is None:
            continue

        try:
            save_video(video_stream, path_to_videos, filename_to_save)
        except ConnectionResetError:
            if os.path.exists(f"{path_to_videos}/{filename_to_save}.mp4"):
                print(f"{path_to_videos}/{filename_to_save}.mp4")
                os.remove(f"{path_to_videos}/{filename_to_save}.mp4")

            print("Connection error")
            return

        save_captions(captions, path_to_captions, filename_to_save)
        total_video_parsed += 1

    with open(path_to_meta_info, "w") as outfile:
        outfile.write(json.dumps(meta_info, indent=4))

    print("total_video_count", total_video_count)
    print("total_video_parsed", total_video_parsed)
