import os
from tqdm import tqdm
import json
import random

from configs import *
from src.utils import extract_audio, create_file, create_directory
from src.audio_to_intervals import get_subtitles_by_intervals, cut_audio_by_intervals

random.seed(4)


if __name__ == "__main__":
    create_directory(path_to_audios)

    videos_list = os.listdir(path_to_videos)
    videos_list = [os.path.join(path_to_videos, video_name) for video_name in videos_list]

    for video_path in tqdm(videos_list):
        extract_audio(video_path, path_to_audios)

    create_directory(path_to_cut_audio_dataset)
    create_file(path_to_cut_audio_dataset_labels)

    create_directory(path_to_cut_audio_dataset_eval)
    create_file(path_to_cut_audio_dataset_eval_labels)

    audios_list = os.listdir(path_to_audios)
    audios_list = [os.path.join(path_to_audios, audio_name) for audio_name in audios_list]

    random.shuffle(audios_list)
    split_point = int(len(audios_list) * train_percent)

    train_audio_list, eval_audio_list = audios_list[:split_point], audios_list[split_point:]

    with open(path_to_meta_info, "r") as f:
        meta_info = json.load(f)

    labels = {}
    labels_eval = {}

    for rec in meta_info:
        captions_path = rec["captions_path"]
        # TODO: need to be reengineered
        path_to_audio = path_to_audios + "/" + captions_path.split("/")[-1][:-4] + ".wav"

        try:
            subs = get_subtitles_by_intervals(captions_path, max_pause=5, max_time_for_sample=30)
            if path_to_audio in train_audio_list:
                cut_audio_by_intervals(path_to_audio, path_to_cut_audio_dataset, subs, labels)

            elif path_to_audio in eval_audio_list:
                cut_audio_by_intervals(path_to_audio, path_to_cut_audio_dataset_eval, subs, labels_eval)
        # fix this
        except Exception as e:
            print(e)
            print(captions_path)
            print(path_to_audio)
            # continue


    with open(path_to_cut_audio_dataset_labels, "a", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
        f.write('\n')

    with open(path_to_cut_audio_dataset_eval_labels, "a", encoding="utf-8") as f:
        json.dump(labels_eval, f, ensure_ascii=False)
        f.write('\n')