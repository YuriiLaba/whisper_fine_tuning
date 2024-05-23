import pandas as pd
from configs import *
from src.utils import read_jsonl_to_dataframe, list_folders, list_wav_files, create_directory
from tqdm import tqdm
import whisper

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def prediction_whisper(audio_path, model):
    result = whisper.transcribe(model, audio_path, language="uk")
    return result['text']


labels = read_jsonl_to_dataframe(path_to_cut_audio_dataset_labels)
dataset_folders = list_folders(path_to_cut_audio_dataset)

model_name = "large"
model = whisper.load_model(model_name, device="cuda")

create_directory(root_path + "results")

for folder_path in tqdm(dataset_folders, total=len(dataset_folders), desc="Processing video"):
    results = {
        "prediction": [],
        "label": []
    }
    wav_list = list_wav_files(folder_path)
    for wav_path in tqdm(wav_list, total=len(wav_list), desc="Processing sample"):
        prediction = prediction_whisper(wav_path, model)
        label = labels[labels['wav_path'].str.contains(wav_path)]["label"].values[0]
        results["prediction"].append(prediction)
        results["label"].append(label)

    pd.DataFrame(results).to_csv(root_path + "results/ " + folder_path.split("/")[-1] + ".csv")

