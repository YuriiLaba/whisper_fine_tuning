import os
import jsonlines
from pathlib import Path

from pydub import AudioSegment

from torch.utils.data import Dataset
from torchaudio import load

from whisper import pad_or_trim, log_mel_spectrogram


def get_audio_length(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        audio_length = len(audio) / 1000  # Convert length to seconds
        return audio_length
    except Exception as e:
        print("Error: ", e)


class AudioDataset(Dataset):
    def __init__(self, data_dir, labels_file, **kwargs):
        self.tokenizer = kwargs.pop("tokenizer")
        self.data_dir = data_dir

        with jsonlines.open(labels_file, 'r') as reader:
          for line in reader:
              self.labels = line
        
        self.walker = self.load_walker()


    def load_walker(self):
        samples = []
        walker = sorted(str(p.stem) for p in Path(self.data_dir).glob("*/*" +  ".wav"))

        for sample in walker:
            
            # print(os.path.join("_".join(sample.split("_")[:2]), sample) + ".wav")
            # print(self.labels)
            if os.path.join(self.data_dir, "_".join(sample.split("_")[:2]), sample) + ".wav" in self.labels.keys():
                samples.append(os.path.join("_".join(sample.split("_")[:2]), sample) + ".wav")
                print("DDD")

        return samples

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, index):
        sample = self.walker[index]
        text = self.labels[os.path.join(self.data_dir, sample)].lower()

        audio_path = os.path.join(self.data_dir, sample)
        # print(get_audio_length(audio_path))
        item, _ = load(audio_path)

        padded_audio = pad_or_trim(item)
        mel_spectrogram = log_mel_spectrogram(padded_audio)

        tokenized_text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        label = tokenized_text[1:] + [self.tokenizer.eot]


        return {
            "mel_spectrogram":mel_spectrogram,
            "dec_input":tokenized_text,
            "label":label,
            "text": text
        }