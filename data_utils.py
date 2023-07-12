import json
import os

import pandas as pd

def load_labels(path_to_labels):
    with open(path_to_labels, 'r') as file:
        for line in file:
            labels = json.loads(line)
    return labels


def clean_dataset(path_to_dataset, path_to_all_predictions='', wer_threshold=1):
    labels = load_labels(os.path.join(path_to_dataset, "labels.jsonl"))
    len_of_data = len(list(labels.keys()))
    all_preds_df = pd.read_csv(os.path.join(path_to_all_predictions, "all_predictions.csv"))

    for bad_chunk in all_preds_df[all_preds_df.wer >= wer_threshold]['audio'].values:
        path_to_bad_chunk = os.path.join(path_to_dataset, '_'.join(bad_chunk.split('_')[:2]), bad_chunk + '.wav')

        try:
            del labels[path_to_bad_chunk]
            os.remove(path_to_bad_chunk)

        except KeyError:
            continue

    with open(os.path.join(path_to_dataset, "labels.jsonl"), 'w') as file:
        json.dump(labels, file)
    print(f'Before cleaning {len_of_data} samples')
    print(f'After cleaning {len(list(labels.keys()))} samples')

