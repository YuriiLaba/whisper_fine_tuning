
# remove empty string (string that containe only whitespace)
# remove all whisper predictions where it is Nan
# remove row if label has less than 10 characters
# delete all samples that have WER >= 1 based on whisper large
# df[df["label"].apply(len) > 3*df["prediction"].apply(len)]
# add tensorboard

import pandas as pd
from utils import read_jsonl_to_dataframe, df_to_jsonl
from utils_analysis import calculate_wer

class DataPreProcessor:
    def __init__(self, path_to_dataset, path_to_predcitions):
        self.dataset = read_jsonl_to_dataframe(path_to_dataset)
        self.prediction = pd.read_csv(path_to_predcitions)
        self.dataset = self.dataset.merge(self.prediction[["prediction", "wav_path"]], how="left", on="wav_path")

    def strip_blank_labels(self):
        self.dataset = self.dataset[self.dataset['label'].str.strip() != '']
        self.dataset = self.dataset.dropna(subset=['label'])

    def remove_nan_predictions(self):
        self.dataset = self.dataset.dropna(subset=['prediction'])
    
    def remove_short_labels(self, thr=10):
         self.dataset = self.dataset[self.dataset["label"].apply(len) >= thr]

    def remove_samples_with_high_wer(self, thr=1):
        self.dataset = self.dataset[self.dataset["wer"] <= thr]

    def filter_labels_by_prediction_length_ratio(self, thr=2):
        self.dataset = self.dataset[~(self.dataset["label"].apply(len) > thr*self.dataset["prediction"].apply(len))]
    
    def filter_labels_by_prediction_word_length_ratio(self, thr=1.4):
        self.dataset = self.dataset[~(self.dataset["label"].apply(lambda x: len(x.split(" "))) > thr*self.dataset["prediction"].apply(lambda x: len(x.split(" "))))]

    def remove_manualy_detected_samples(self, samples):
        if len(samples) != 0:
            self.dataset = self.dataset[~(self.dataset["wav_path"].isin(samples))]

    def run(self, clean_dataset_path):
        self.remove_manualy_detected_samples(["dataset/toronto_3/toronto_3_39.wav"])
        self.strip_blank_labels()
        self.remove_nan_predictions()
        self.remove_short_labels()
        
        calculate_wer(self.dataset)

        self.remove_samples_with_high_wer()
        self.filter_labels_by_prediction_length_ratio()
        self.filter_labels_by_prediction_word_length_ratio()

        self.remove_short_labels()


        # print(self.dataset.head())
        df_to_jsonl(self.dataset[["wav_path", "label"]], clean_dataset_path)


# TODO clean not ukrainian text
# TODO є проблема що в евалі є звуки які детактятсья як грати песик... ми думаємо що є в трейні випадки коли є траскрипт звуку як грати песик дужка...
if __name__ == "__main__":
    data_pre_processor = DataPreProcessor("dataset/labels.jsonl", "results/predictions.csv")
    data_pre_processor.run("results/filtered_labels_train.jsonl")

    data_pre_processor = DataPreProcessor("eval_dataset/labels_eval.jsonl", "results/predictions.csv")
    data_pre_processor.run("results/filtered_labels_eval.jsonl")

        
