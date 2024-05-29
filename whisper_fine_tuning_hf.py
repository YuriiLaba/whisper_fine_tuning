import torch
import librosa

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.integrations import NeptuneCallback

import os
import jsonlines
from torch.utils.data import Dataset
from torchaudio import load
from pathlib import Path

import evaluate
import neptune

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Ukrainian", task="transcribe")

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Ukrainian", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.max_length = 500

max_label_length = model.config.max_length

MAX_DURATION_IN_SECONDS = 30.0
max_input_length = MAX_DURATION_IN_SECONDS * 16000

class AudioDataset(Dataset):
    def __init__(self, data_dir, labels_file, **kwargs):
        self.data_dir = data_dir

        with jsonlines.open(labels_file, 'r') as reader:
            for line in reader:
                self.labels = line

        self.walker = self.load_walker()

    def load_walker(self):
        samples = []
        walker = sorted(str(p.stem) for p in Path(self.data_dir).glob("*/*" +  ".wav"))

        for sample in walker:
            if os.path.join(self.data_dir, "_".join(sample.split("_")[:2]), sample) + ".wav" in self.labels.keys():
                samples.append(os.path.join("_".join(sample.split("_")[:2]), sample) + ".wav")
        return samples

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, index):
        sample = self.walker[index]
        text = self.labels[os.path.join(self.data_dir, sample)].lower()

        audio_path = os.path.join(self.data_dir, sample)
        audio, sr = librosa.load(audio_path)

        input_features = feature_extractor(audio, sampling_rate=16_000).input_features[0]
        labels = tokenizer(text).input_ids

        return {
            "input_features": input_features,
            "input_length": len(input_features),
            "labels": labels,
            "labels_length": len(labels)
        }


path_to_dataset = ""

train_root_dir = os.path.join(path_to_dataset, "dataset/")
train_labels_file = os.path.join(path_to_dataset, "results/filtered_labels_train.jsonl")

eval_root_dir = os.path.join(path_to_dataset, "eval_dataset/")
eval_labels_file = os.path.join(path_to_dataset, "results/filtered_labels_eval.jsonl")


train_dataset = AudioDataset(train_root_dir, train_labels_file)
eval_dataset = AudioDataset(eval_root_dir, eval_labels_file)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=20_000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1_000,
    eval_steps=200,
    logging_steps=1,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

run = neptune.init_run(project="vova.mudruy/Toronto-whisper",
                       api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlYTg0NWQxYy0zNTVkLTQwZDktODJhZC00ZjgxNGNhODE2OTIifQ==")

neptune_callback = NeptuneCallback(run=run, log_parameters=False)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[neptune_callback]
)

trainer.train()