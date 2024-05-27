import os
import pandas as pd
from collections import Counter
import string
import re
import librosa
from tqdm import tqdm
from jiwer import wer
import plotly.express as px


def remove_punctuation(text):
    """
    Removes all punctuation from a given string.
    """

    ACUTE = chr(0x301)
    GRAVE = chr(0x300)

    if type(text) == str:
        text = text.replace("#@)₴?$0", '')
        punctuations = string.punctuation + '«»' + "''’" + '–' + '₴' + '…' + '’' + ACUTE + GRAVE
        translator = str.maketrans('', '', punctuations)
        text = text.translate(translator)

        text = " ".join(text.split())
        if len(text) == 0:
            return ' '
        return text

    else:
        return str(text)


def remove_text_in_brackets(text):
    result = re.sub(r'\([^)]*\)', '', text)
    result = re.sub(r'\s{2,}', ' ', result)
    return result


def clean_text_before_wer(text):
    text = text.lower()
    text = remove_text_in_brackets(text)
    return remove_punctuation(text)


def custom_wer(x, y):
    try:
        return wer(x, y)
    except ValueError:
        return 1


def calculate_wer(prediction_df):
    if ("clean_label" not in prediction_df.columns) or ("clean_prediction" not in prediction_df.columns):
        prediction_df['clean_label'] = prediction_df['label'].apply(clean_text_before_wer)
        prediction_df['clean_prediction'] = prediction_df['prediction'].apply(clean_text_before_wer)

    prediction_df['wer'] = prediction_df.apply(lambda x: custom_wer(x['clean_label'], x['clean_prediction']), axis=1)
    # return prediction_df['wer'].mean(), prediction_df['wer'].median()


def wer_for_dataset(dataset_path):
    wer_by_videos = []
    dataframes = []

    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(dataset_path, file_name)
            df = pd.read_csv(file_path)
            mean_wer, median_wer = calculate_wer(df)

            wer_by_videos.append((file_name, mean_wer, median_wer))

            dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df.wer.mean(), wer_by_videos, combined_df


def plot_wer_distribution(wer_by_videos, aggregation='mean'):
    names = [item[0] for item in wer_by_videos]
    if aggregation == 'median':
        values = [item[2] for item in wer_by_videos]
    else:
        values = [item[1] for item in wer_by_videos]

    fig = px.line(x=names, y=values, markers=True)
    fig.show()


def common_words(path_to_data, most_common=30):
    folder_path = os.path.join(path_to_data, 'dataset/results')
    df_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, filename))
            df_list.append(df)

    wer_results = pd.concat(df_list, ignore_index=True)
    print('WER = ', wer_results.wer.mean())

    diff = wer_results.apply(lambda row: set(row['label'].split()) - set(row['prediction'].split()), axis=1)
    words = [word for sublist in diff for word in sublist if len(word) > 2]
    word_counts = Counter(words)

    most_common_words = word_counts.most_common(most_common)
    for element, count in most_common_words:
        print(f'{element}: {count}')


def calculate_duration(dataset_path):
    total_duration = 0
    subfolder_durations = {}

    for root, dirs, files in tqdm(os.walk(dataset_path), total=len(os.listdir(dataset_path))):
        subfolder_duration = 0

        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(('.mp3', '.wav', '.ogg')):
                try:
                    audio, sr = librosa.load(file_path)
                    duration = librosa.get_duration(y=audio, sr=sr)
                    subfolder_duration += duration
                except Exception as e:
                    print(f"Error processing file: {file_path}\n{e}")

        subfolder_durations[root] = subfolder_duration
        total_duration += subfolder_duration

    return total_duration, subfolder_durations
