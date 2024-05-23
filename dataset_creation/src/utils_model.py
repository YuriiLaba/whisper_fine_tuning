import os
import pandas as pd

from tqdm import tqdm

import torch
from torchaudio import load
import whisper
from whisper import pad_or_trim, log_mel_spectrogram, DecodingOptions

from utils_analysis import clean_text_before_wer


def load_model(model_size, model_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = whisper.load_model(model_size, device=device)

    if model_path is not None:
        if device == 'cuda':
            params = torch.load(model_path, map_location='cuda:0')
        else:
            params = torch.load(model_path)
        model.load_state_dict(params)

    return model


def predictions_to_df(predictions):
    df = pd.DataFrame({'label': [i[0] for i in predictions],
                       'prediction': [i[1] for i in predictions]
                       })

    df['clean_label'] = df['label'].apply(clean_text_before_wer)
    df['clean_prediction'] = df['prediction'].apply(clean_text_before_wer)
    return df


def predict_finetune(audio_path, model):
    options = DecodingOptions(language="uk", without_timestamps=True, fp16=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    item, _ = load(audio_path)

    padded_audio = pad_or_trim(item)
    mel_spectrogram = log_mel_spectrogram(padded_audio)
    res = model.decode(mel_spectrogram.to(device), options)
    return res[0].text


def predict_video(video_id, dataset_path, model, labels, pred_method='base'):
    results_one_video = []

    video_folder = f"toronto_{video_id}"
    path_to_video = os.path.join(dataset_path, video_folder)

    total_audios = len(os.listdir(path_to_video))

    for i in tqdm(range(total_audios), leave=False):
        audio_path = os.path.join(path_to_video, f"toronto_{video_id}_{i}.wav")

        try:
            true_label = labels[audio_path]
        except KeyError:
            true_label = 'missed audio'
            results_one_video.append((true_label, 'missed audio'))
            continue

        if pred_method == 'base':
            result = model.transcribe(audio_path)['text'] # language = 'uk'
        elif pred_method == 'fine-tune':
            result = predict_finetune(audio_path, model)
        else:
            break

        results_one_video.append((true_label, result))
    return predictions_to_df(results_one_video)
