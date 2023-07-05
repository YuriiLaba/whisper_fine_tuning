import os

import torch

import whisper
from whisper.tokenizer import get_tokenizer

from fine_tune_whisper import Trainer
from toronto_dataset import AudioDataset

path_to_dataset = "/content/drive/MyDrive/ukrainian-youtube-stt-dataset/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = whisper.load_model("medium")
tokenizer = get_tokenizer(model.is_multilingual, language="uk", task="transcribe")

train_root_dir = os.path.join(path_to_dataset, "dataset/")
train_labels_file = os.path.join(path_to_dataset, "dataset/labels.jsonl")
eval_root_dir = os.path.join(path_to_dataset, "eval_dataset/")
eval_labels_file = os.path.join(path_to_dataset, "eval_dataset/labels_eval.jsonl")
train_dataset = AudioDataset(train_root_dir, train_labels_file, tokenizer=tokenizer)
eval_dataset = AudioDataset(eval_root_dir, eval_labels_file, tokenizer=tokenizer)

model_params = {
    "n_epochs": 4,
    "batch_size": 4,
    "learning_rate": 1e-5,
    "device": device
}

trainer = Trainer(model, train_dataset, eval_dataset, ".", model_params)
trainer.train()