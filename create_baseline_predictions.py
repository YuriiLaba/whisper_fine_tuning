import sys
sys.path.append("/home/laba/golos/whisper_fine_tuning/dataset_creation/src/")

import os
import pandas as pd
from tqdm import tqdm
import librosa
import glob

from utils import load_labels
from utils_model import load_model, predict_video

model_large = load_model('large')

def create_baseline_predictions(path_to_data, label_name, model):
    labels = load_labels(os.path.join(path_to_data, label_name))
    data = [i for i in glob.glob(os.path.join(path_to_data, '*')) if not os.path.isfile(i)]

    predictions = []

    for audio_folder in tqdm(data, desc="Total audio files", leave=True):
        predicted_video = predict_video(video_id=audio_folder.split('_')[-1],
                                        dataset_path=path_to_data,
                                        model=model,
                                        labels=labels, 
                                        do_clean=False)

        predicted_video["audio_folder"] = audio_folder + "/" + predicted_video.index.astype(str)
        predictions.append(predicted_video)
    
    pd.concat(predictions).to_csv(f"results/{label_name.split('.')[0]}_baseline_prediction.csv", index=False)

create_baseline_predictions("eval_dataset", "labels_eval.jsonl", model_large)
create_baseline_predictions("dataset", "labels.jsonl", model_large)