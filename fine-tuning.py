import os
import random

import numpy as np

import torch

import whisper
from whisper.tokenizer import get_tokenizer

import neptune

from fine_tune_whisper import Trainer
from toronto_dataset import AudioDataset
from data_utils import clean_dataset, remove_absolute_path


use_multiple_gpu = False
path_to_dataset = ""

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


run = neptune.init_run(project="vova.mudruy/Toronto-whisper",
                       api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlYTg0NWQxYy0zNTVkLTQwZDktODJhZC00ZjgxNGNhODE2OTIifQ==")

model_params = {
    "n_epochs": 4,
    "batch_size": 4,
    "learning_rate": 1e-5,
    "early_stopping": 50,
    "calc_val_num": 300,
    "model_size": "small",
}

remove_absolute_path()

model = whisper.load_model(model_params['model_size'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = get_tokenizer(model.is_multilingual, language="uk", task="transcribe")

train_root_dir = os.path.join(path_to_dataset, "dataset/")
train_labels_file = os.path.join(path_to_dataset, "dataset/labels.jsonl")
eval_root_dir = os.path.join(path_to_dataset, "eval_dataset/")
eval_labels_file = os.path.join(path_to_dataset, "eval_dataset/labels_eval.jsonl")

clean_dataset(os.path.join(path_to_dataset, 'dataset'), label_file = "labels.jsonl", wer_threshold=0.8)
clean_dataset(os.path.join(path_to_dataset, 'eval_dataset'), label_file = "labels_eval.jsonl", wer_threshold=0.8)

train_dataset = AudioDataset(train_root_dir, train_labels_file, tokenizer=tokenizer)
eval_dataset = AudioDataset(eval_root_dir, eval_labels_file, tokenizer=tokenizer)

run["parameters"] = model_params

model_params['device'] = device

trainer = Trainer(model, train_dataset, eval_dataset, ".", model_params, run)
trainer.train()
run.stop()