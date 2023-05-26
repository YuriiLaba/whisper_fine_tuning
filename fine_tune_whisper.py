import torch
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH
import torch.optim as optim
import torch.nn as nn
import tqdm
from pathlib import Path

import whisper.whisper
from whisper.whisper import load_model, pad_or_trim, log_mel_spectrogram
from whisper.whisper.tokenizer import get_tokenizer

from asr_metrics import wer


def collate_fn(items):
    n_batch = len(items)
    _, n_mel, n_frame = items[0]["mel_spectrogram"].shape
    text_list, label_len, dec_input_len = [], [], []

    for item in items:
        text_list.append(item["text"])
        label_len.append(len(item["label"]))
        dec_input_len.append(len(item["dec_input"]))

    max_label_len = max(label_len + dec_input_len)

    batch_mel = torch.zeros(n_batch, n_mel, n_frame)
    batch_label = torch.full([n_batch, max_label_len], fill_value=-100, dtype=torch.long)
    batch_dec_input = torch.full([n_batch, max_label_len], fill_value=50257, dtype=torch.long)

    for idx, item in enumerate(items):
        n_frame = item["mel_spectrogram"].shape[-1]
        batch_mel[idx, :, :n_frame] = item["mel_spectrogram"]
        batch_label[idx, :label_len[idx]] = torch.tensor(item["label"], dtype=torch.long)
        batch_dec_input[idx, :dec_input_len[idx]] = torch.tensor(item["dec_input"], dtype=torch.long)

    return {
        "mel_spectrogram": batch_mel,
        "dec_input": batch_dec_input,
        "label": batch_label,
        "text": text_list
    }


class MyDataset(LIBRISPEECH):
    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs.pop("tokenizer")

        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        padded_audio = pad_or_trim(item[0])
        mel_spectrogram = log_mel_spectrogram(padded_audio)
        text = item[2].lower()

        tokenized_text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        label = tokenized_text[1:] + [self.tokenizer.eot]

        return {
            "mel_spectrogram":mel_spectrogram,
            "dec_input":tokenized_text,
            "label":label,
            "text": text
        }


class Trainer:
    def __init__(self, model, train_dataset, eval_dataset, output_dir, model_params):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.model_params = model_params

        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=model_params["batch_size"], collate_fn=collate_fn)
        self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=model_params["batch_size"], collate_fn=collate_fn)

        self.options = whisper.whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=model_params["learning_rate"])
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def _get_ckpt_path(self, epoch, iter):
        return self.output_dir.joinpath(f'ckpt_epoch_{epoch}_iter_{iter}.pt')

    def predict(self, mel_spectrogram):
        with torch.no_grad():
            res = self.model.decode(mel_spectrogram, self.options)
        return [item.text for item in res]

    def train_step(self, batch):
        model_output = self.model(batch["mel_spectrogram"].to(self.model_params["device"]),  batch["dec_input"].to(self.model_params["device"]))
        loss = self.criterion(model_output.view(-1, model_output.shape[-1]), target=batch["label"].view(-1).to(self.model_params["device"]))
        return loss

    def train_epoch(self, epoch):
        for idx, batch in enumerate(tqdm.tqdm(self.train_dataloader)):
            self.optimizer.zero_grad()
            loss = self.train_step(batch)
            loss.backward()
            self.optimizer.step()

            if idx % 100 == 0:
                print(f'epoch {epoch}, iter {idx:05}: x-entropy={loss.item():.3f}')
            if idx % 500 == 0 and idx != 0:
                torch.save(self.model.state_dict(), self._get_ckpt_path(epoch, idx))

    def validate(self, epoch):
        val_wer = []
        for idx, batch in enumerate(tqdm.tqdm(self.eval_dataloader)):
            target_text = batch["text"]
            predicted_text = self.predict(batch["mel_spectrogram"].to(self.model_params["device"]))

            for target_text_sample, predicted_text_sample in zip(target_text, predicted_text):
                val_wer.append(wer(target_text_sample.lower(), predicted_text_sample.lower()))

            # calculate wer only on the first batch
            break

        mean_wer = sum(val_wer)/len(val_wer)
        print(f'epoch {epoch}. Validation WER: {mean_wer:.3f}')

    def train(self):
        for e in range(self.model_params["n_epochs"]):
            self.validate(e-1)
            self.train_epoch(e)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = torch.load("/content/drive/MyDrive/Colab Notebooks/language-recognition-school/making-sense-of-speech/patriotic_whisper_mixed_en_uk.pt")
    model = load_model("tiny", device=device)
    model.load_state_dict(params)

    tokenizer = get_tokenizer(model.is_multilingual, language="en", task="transcribe")

    train_dataset = MyDataset(root=".", url='train-clean-100', download=True, tokenizer=tokenizer)
    eval_dataset = MyDataset(root=".", url='dev-clean', download=True, tokenizer=tokenizer)

    model_params = {
        "n_epochs": 2,
        "batch_size": 16,
        "learning_rate": 1e-5,
        "device": device
    }

    trainer = Trainer(model, train_dataset, eval_dataset, ".", model_params)
    trainer.train()

