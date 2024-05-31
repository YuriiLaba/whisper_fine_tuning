import os
import math
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from whisper import DecodingOptions
import whisper

from jiwer import wer
from wer_utils import clean_text_before_wer

import gc

torch.backends.cudnn.benchmark = True

def report_gpu():
    torch.cuda.empty_cache()
    gc.collect()

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def collate_fn(items):
    n_batch = len(items)
    # _, n_mel, n_frame = items[0]["mel_spectrogram"].shape
    n_mel, n_frame = items[0]["mel_spectrogram"].shape
    text_list, label_len, dec_input_len, audio_path_list = [], [], [], []

    for item in items:
        text_list.append(item["text"])
        label_len.append(len(item["label"]))
        dec_input_len.append(len(item["dec_input"]))
        audio_path_list.append(item["audio_path"])

    max_label_len = max(label_len + dec_input_len)

    batch_mel = torch.zeros(n_batch, n_mel, n_frame)
    batch_label = torch.full([n_batch, max_label_len], fill_value=-100, dtype=torch.long)
    batch_dec_input = torch.full([n_batch, max_label_len], fill_value=50257, dtype=torch.long)

    # opportunity to optimize this part of code
    for idx, item in enumerate(items):
        n_frame = item["mel_spectrogram"].shape[-1]
        batch_mel[idx, :, :n_frame] = item["mel_spectrogram"]
        batch_label[idx, :label_len[idx]] = torch.tensor(item["label"], dtype=torch.long)
        batch_dec_input[idx, :dec_input_len[idx]] = torch.tensor(item["dec_input"], dtype=torch.long)

    return {
        "mel_spectrogram": batch_mel,
        "dec_input": batch_dec_input,
        "label": batch_label,
        "text": text_list,
        "audio_path": audio_path_list
    }


class Trainer:
    def __init__(self, model, train_dataset, eval_dataset, output_dir, model_params, run, debug=False):
        self.model = nn.DataParallel(model)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.model_params = model_params

        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=model_params["batch_size_train"],
                                           collate_fn=collate_fn)
        self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=model_params["batch_size_eval"],
                                          collate_fn=collate_fn)

        self.options = DecodingOptions(language="uk", without_timestamps=True, fp16=False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=model_params["learning_rate"])
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.neptune_logger = run

        self.early_stop = model_params['early_stopping']

        self.calc_val_num = model_params['calc_val_num']
        self.best_mean_wer = 1000
        self.best_epoch_mean_wer = 1000
        self.mean_wer = None
        self.mean_wer_clean = None
        self.bad_rounds = 0
        self.debug = debug

    def _get_ckpt_path(self, epoch, iteration):
        return self.output_dir.joinpath(f'ckpt_epoch_{epoch}_iter_{iteration}.pt')

    def predict(self, mel_spectrogram):
        with torch.no_grad():
            res = self.model.module.decode(mel_spectrogram, self.options)
        return [item.text for item in res]

    def train_step(self, batch):
        model_output = self.model(batch["mel_spectrogram"].to(self.model_params["device"]),  batch["dec_input"].to(self.model_params["device"]))
        loss = self.criterion(model_output.view(-1, model_output.shape[-1]), target=batch["label"].view(-1).to(self.model_params["device"]))
        return loss

    def train_epoch(self, epoch, loss_metrics):
        self.model.train()

        train_bar = tqdm(self.train_dataloader, desc='Train')
        
        if self.debug:
            print("Free memory at the beggining of train epoch:", convert_size(torch.cuda.mem_get_info()[0]))
            print("Total memory:", convert_size(torch.cuda.mem_get_info()[1]))
        
        for idx, batch in enumerate(train_bar):
            
            self.optimizer.zero_grad()
            if self.debug:
                print("Free memory before the train step:", convert_size(torch.cuda.mem_get_info()[0]))
            loss = self.train_step(batch)
            if self.debug:
                print("Free memory after the train step:", convert_size(torch.cuda.mem_get_info()[0]))
            loss.backward()
            self.optimizer.step()

            loss_metrics.update(loss.detach().cpu().numpy(), self.model_params["batch_size_train"])
            train_bar.set_postfix(loss=loss_metrics.avg, epoch=epoch, step=idx)

            del loss
            del batch

            if self.debug:
                print("Free memory after the empty_cache:", convert_size(torch.cuda.mem_get_info()[0]))

            if self.neptune_logger is not None:
                self.neptune_logger["train/loss"].append(loss_metrics.avg)

            if (self.calc_val_num is not None ) and (idx % self.calc_val_num == 0) and (idx != 0):
                self.validate(epoch - 1, step = idx, subsample=True)
                self.model.train()

                if self.mean_wer < self.best_mean_wer:
                    self.best_mean_wer = self.mean_wer
                    self.bad_rounds = 0
                    torch.save(self.model.module.state_dict(), self._get_ckpt_path(epoch, idx))
                else:
                    self.bad_rounds += 1

                if self.bad_rounds == self.early_stop:
                    print(f'Early stopping detected, Best WER was {self.best_mean_wer:.3f} at {epoch-self.bad_rounds}. Current WER = {self.mean_wer:.3f}')
                    return None
            

    def validate(self, epoch, step = None, subsample = False):
        self.model.eval()

        val_wer = []
        val_wer_clean = []

        eval_bar = tqdm(self.eval_dataloader, desc='Eval', leave=False)
        for idx, batch in enumerate(eval_bar):
            target_text = batch["text"]
            predicted_text = self.predict(batch["mel_spectrogram"].to(self.model_params["device"]))
            # transcribe_text = whisper.transcribe(self.model, batch['audio_path'][0])

            # audio = whisper.load_audio(batch['audio_path'][0])
            # audio = whisper.pad_or_trim(audio)
            # mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            # _, probs = self.model.detect_language(mel)
            # print(f"Detected language: {max(probs, key=probs.get)}")
            # options = whisper.DecodingOptions()
            # result = whisper.decode(self.model, mel, options)

            for target_text_sample, predicted_text_sample in zip(target_text, predicted_text):
                
                val_wer.append(wer(target_text_sample.lower(),
                                   predicted_text_sample.lower()))
                try:
                    val_wer_clean.append(wer(clean_text_before_wer(target_text_sample),
                                             clean_text_before_wer(predicted_text_sample)))
                except ValueError:
                    val_wer_clean.append(1)

            if idx == 50 and subsample:
                break
            del batch
            torch.cuda.empty_cache()

        # print(next(iter(eval_bar)))
        self.mean_wer = sum(val_wer)/len(val_wer)
        self.mean_wer_clean = sum(val_wer_clean) / len(val_wer)

        if subsample:
            print(f'epoch {epoch}, Step {step}. Subsample validation WER: {self.mean_wer:.3f} Clean WER: {self.mean_wer_clean:.3f}')
            if self.neptune_logger is not None:
                self.neptune_logger["val/WER_subsample"].append(self.mean_wer)
                self.neptune_logger["val/WER_clean_subsample"].append(self.mean_wer_clean)
        else:
            print(f'epoch {epoch}. Validation WER: {self.mean_wer:.3f} Clean WER: {self.mean_wer_clean:.3f}')
            if self.neptune_logger is not None:
                self.neptune_logger["val/WER"].append(self.mean_wer)
                self.neptune_logger["val/WER_clean"].append(self.mean_wer_clean)

    def train(self):
        loss_metrics = AverageMeter()

        for e in range(self.model_params["n_epochs"]):
            self.train_epoch(e, loss_metrics)
            report_gpu()

            self.validate(e - 1)
            report_gpu()

            if self.mean_wer < self.best_epoch_mean_wer:
                self.best_epoch_mean_wer = self.mean_wer
                torch.save(self.model.module.state_dict(), self._get_ckpt_path('best_epoch', ''))

        torch.save(self.model.module.state_dict(), self._get_ckpt_path('final', ''))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=None):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window_size = window_size

    def reset(self):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.window_size and (self.count >= self.window_size):
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count