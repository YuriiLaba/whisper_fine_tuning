import os
import subprocess
import pandas as pd
import jsonlines
import json


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_file(path):
    if not os.path.exists(path):
        os.system(f'touch {path}')


def extract_audio(path_to_video, path_to_audios, output_ext="wav"):
    filename = path_to_video.split("/")[-1][:-4]
    subprocess.call(["ffmpeg", "-i", path_to_video, "-ac", "1", "-c:v", "copy",
                     f"{path_to_audios}/{filename}.{output_ext}"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def read_jsonl_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = jsonlines.Reader(f)
        for obj in reader:
            data.append(obj)

    return pd.DataFrame(data).T.reset_index().rename(columns={"index": "wav_path", 0: "label"})


def list_folders(directory):
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folders.append(item_path)
    return folders


def list_wav_files(path_to_dir):
    wav_files = []
    for root, _, files in os.walk(path_to_dir):
        for f in files:
            if f.endswith(".wav"):
                file_path = os.path.join(root, f)
                wav_files.append(file_path)
    return wav_files


def load_labels(path_to_labels):
    with open(path_to_labels, 'r') as file:
        for line in file:
            labels = json.loads(line)
    return labels

def df_to_jsonl(df, output_path):
    df.set_index("wav_path", inplace=True)
    
    with open(output_path, 'w', encoding="utf-8") as file:
        json.dump(df["label"].to_dict(), file, ensure_ascii=False)
