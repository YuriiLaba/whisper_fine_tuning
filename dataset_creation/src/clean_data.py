import re
import string



ACUTE = chr(0x301)
GRAVE = chr(0x300)


def clean_sentence(sentence):
    sentence = sentence.replace("«#@)₴?$0»", 'Ґрати песик дужка гривня знак питання долар нуль').strip()
    sentence = sentence.replace("#@)₴?$0", 'Ґрати песик дужка гривня знак питання долар нуль').strip()

    sentence = sentence.replace("$0", '').strip()
    sentence = sentence.replace("ヴィチギョーザ。", '').strip()
    sentence = sentence.replace("何か新しいことに挑戦します", '').strip()
    sentence = sentence.replace("№", ' номер ').strip()

    sentence = re.sub(r'\([^)]*\)', '', sentence)
    sentence = re.sub(r'\[.*?\]', '', sentence)

    punctuations = string.punctuation + '«»' + "''" + '–' + '₴' + '…' + '“' + '“' + '—' + '”'
    translator = str.maketrans(punctuations, ' ' * len(punctuations), ACUTE + GRAVE)
    sentence = sentence.translate(translator)

    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
