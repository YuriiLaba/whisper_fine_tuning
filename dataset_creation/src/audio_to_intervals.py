from pydub import AudioSegment
import pysrt
import os
from tqdm import tqdm

from src.clean_data import clean_sentence


def get_subtitles(filename):
    subs = pysrt.open(filename)
    time_intervals = []
    for sub in subs:
        sub_text = sub.text.replace("\n", " ")
        sub_start = sub.start
        sub_end = sub.end
        time_intervals.append((sub_text, sub_start, sub_end))
    return time_intervals


def get_subtitles_by_intervals(filename, max_pause, max_time_for_sample):
    subs = pysrt.open(filename)
    time_intervals = []
    start_time = subs[0].start
    end_time = subs[0].end

    sub_text = subs[0].text.replace("\n", " ")

    for sub in subs[1:]:
        if abs((sub.start - end_time).ordinal/1000) > max_pause:
            # if there is a long time distance between captions, we don't
            # want to store such pause (music, etc.) In future we should
            # add human speech detector

            time_intervals.append((sub_text.strip(), start_time, end_time))
            start_time = sub.start
            end_time = sub.end
            sub_text = sub.text.replace("\n", " ")
            continue

        total_time = abs((sub.end - start_time).ordinal/1000)
        total_time += abs((sub.end - start_time).ordinal/1000)

        if total_time <= max_time_for_sample:
            end_time = sub.end
            sub_text += " " + sub.text.replace("\n", " ")
        else:
            time_intervals.append((sub_text.strip(), start_time, end_time))
            start_time = sub.start
            end_time = sub.end
            sub_text = sub.text.replace("\n", " ")

    time_intervals.append((sub_text.strip(), start_time, end_time))

    return time_intervals


def cut_audio_by_intervals(path_to_audio, path_to_cut_audio, subs, labels):
    audio = AudioSegment.from_file(path_to_audio)

    for i, interval in enumerate(tqdm(subs)):

        output_filename = f"{path_to_cut_audio}/{path_to_audio.split('/')[-1][:-4]}/{path_to_audio.split('/')[-1][:-4]}_{i}.wav"

        if os.path.exists(output_filename):
            continue

        if not os.path.exists(f"{path_to_cut_audio}/{path_to_audio.split('/')[-1][:-4]}"):
            os.mkdir(f"{path_to_cut_audio}/{path_to_audio.split('/')[-1][:-4]}")

        start_time, end_time = interval[1], interval[2]
        text = clean_sentence(interval[0])

        start_ms = start_time.hours * 3600000 + start_time.minutes * 60000 + start_time.seconds * 1000 + start_time.milliseconds
        end_ms = end_time.hours * 3600000 + end_time.minutes * 60000 + end_time.seconds * 1000 + end_time.milliseconds

        interval_audio = audio[start_ms:end_ms]

        interval_audio.export(output_filename, format="wav")
        labels[output_filename] = text
