import json
import os
import re

import pandas as pd

def load_labels(path_to_labels):
    with open(path_to_labels, 'r') as file:
        for line in file:
            labels = json.loads(line)
    return labels


def replace_in_dict_keys(dictionary, old_str, new_str):
    new_dict = {}
    for key, value in dictionary.items():
        new_key = key.replace(old_str, new_str)
        new_dict[new_key] = value
    return new_dict


def remove_absolute_path():
    train_labels = load_labels("dataset/labels.jsonl")
    eval_labels = load_labels("eval_dataset/labels_eval.jsonl")

    train_labels = replace_in_dict_keys(train_labels, '/content/drive/MyDrive/ukrainian-youtube-stt-dataset/', '')
    eval_labels = replace_in_dict_keys(eval_labels, '/content/drive/MyDrive/ukrainian-youtube-stt-dataset/', '')

    with open("dataset/labels.jsonl", 'w', encoding="utf-8") as file:
        json.dump(train_labels, file, ensure_ascii=False)

    with open("eval_dataset/labels_eval.jsonl", 'w', encoding="utf-8") as file:
        json.dump(eval_labels, file, ensure_ascii=False)


def clean_dataset(path_to_dataset, path_to_all_predictions='', wer_threshold=1):
    labels = load_labels(os.path.join(path_to_dataset, "labels.jsonl"))
    len_of_data = len(list(labels.keys()))
    all_preds_df = pd.read_csv(os.path.join(path_to_all_predictions, "all_predictions.csv"))

    for bad_chunk in all_preds_df[(all_preds_df.wer >= wer_threshold) | (all_preds_df.clean_label == ' ')]['audio'].values:
        path_to_bad_chunk = os.path.join(path_to_dataset, '_'.join(bad_chunk.split('_')[:2]), bad_chunk + '.wav')

        try:
            del labels[path_to_bad_chunk]
            os.remove(path_to_bad_chunk)

        except KeyError:
            continue

        except FileNotFoundError:
            print('file already deleted')
            continue

    for key, value in labels.items():
        labels[key] = re.sub(r'\([^)]*\)', '', value)

    with open(os.path.join(path_to_dataset, "labels.jsonl"), 'w', encoding="utf-8") as file:
        json.dump(labels, file, ensure_ascii=False)

    print(f'Before cleaning {len_of_data} samples')
    print(f'After cleaning {len(list(labels.keys()))} samples')