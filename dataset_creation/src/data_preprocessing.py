import pandas as pd
import re
from utils import read_jsonl_to_dataframe, df_to_jsonl
from utils_analysis import calculate_wer

class DataPreProcessor:
    def __init__(self, path_to_dataset, path_to_predcitions, debug=False):
        self.dataset = read_jsonl_to_dataframe(path_to_dataset)
        self.prediction = pd.read_csv(path_to_predcitions)
        self.dataset = self.dataset.merge(self.prediction[["prediction", "wav_path"]], how="left", on="wav_path")
        self.debug=debug

    def strip_blank_labels(self):
        self.dataset = self.dataset[self.dataset['label'].str.strip() != '']
        self.dataset = self.dataset.dropna(subset=['label'])
        if self.debug:
            print('Size after removing blank labels = ', len(self.dataset))

    def remove_nan_predictions(self):
        self.dataset = self.dataset.dropna(subset=['prediction'])
        if self.debug:
            print('Size after removing nan predictions = ', len(self.dataset))
    
    def remove_short_labels(self, thr=10):
        self.dataset = self.dataset[self.dataset["label"].apply(len) >= thr]
        if self.debug:
            print('Size after removing short labels = ', len(self.dataset))

    def remove_samples_with_high_wer(self, thr=1):
        self.dataset = self.dataset[self.dataset["wer"] <= thr]
        if self.debug:
            print('Size after removing high WER = ', len(self.dataset))

    def filter_labels_by_prediction_length_ratio(self, thr=2):
        self.dataset = self.dataset[~(self.dataset["label"].apply(len) > thr*self.dataset["prediction"].apply(len))]
        if self.debug:
            print('Size after removing by lenght ratio = ', len(self.dataset))
    
    def filter_labels_by_prediction_word_length_ratio(self, thr=1.4):
        self.dataset = self.dataset[~(self.dataset["label"].apply(lambda x: len(x.split(" "))) > thr*self.dataset["prediction"].apply(lambda x: len(x.split(" "))))]
        if self.debug:
            print('Size after removing by word legnth ratio = ', len(self.dataset))

    def remove_samples_with_invalid_characters(self):
        combined_text = ''.join(self.dataset['label'].astype(str))
        ukr_eng_pattern = r'[\u0400-\u04FFa-zA-Z0-9 ’ʼ‘\s]+'
        
        non_ukr_eng_chars = {char for char in combined_text if not re.match(ukr_eng_pattern, char)}

        def contains_non_ukr_eng(label):
            return any(char in non_ukr_eng_chars for char in label)
        
        self.dataset = self.dataset[~self.dataset['label'].apply(contains_non_ukr_eng)]
        if self.debug:
            print('Size after removing samples with invalid characters = ', len(self.dataset))

    def remove_manualy_detected_samples(self, samples):
        if len(samples) != 0:
            self.dataset = self.dataset[~(self.dataset["wav_path"].isin(samples))]
        if self.debug:
            print('Size after removing manually detected errors = ', len(self.dataset))

    def run(self, clean_dataset_path):
        if self.debug:
            print('Initial size = ', len(self.dataset))

        self.remove_manualy_detected_samples(["dataset/toronto_3/toronto_3_39.wav"])
        self.strip_blank_labels()
        self.remove_nan_predictions()
        self.remove_short_labels()
        
        calculate_wer(self.dataset)

        self.remove_samples_with_high_wer()
        self.filter_labels_by_prediction_length_ratio()
        self.filter_labels_by_prediction_word_length_ratio()

        self.remove_short_labels()

        self.remove_samples_with_invalid_characters()

        df_to_jsonl(self.dataset[["wav_path", "label"]], clean_dataset_path)


# TODO clean not ukrainian text
# TODO є проблема що в евалі є звуки які детактятсья як грати песик... ми думаємо що є в трейні випадки коли є траскрипт звуку як грати песик дужка...
if __name__ == "__main__":
    data_pre_processor = DataPreProcessor("dataset/labels.jsonl", "results/predictions.csv", debug=True)
    data_pre_processor.run("results/filtered_labels_train.jsonl")

    data_pre_processor = DataPreProcessor("eval_dataset/labels_eval.jsonl", "results/predictions.csv", debug=True)
    data_pre_processor.run("results/filtered_labels_eval.jsonl")
