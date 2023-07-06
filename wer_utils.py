import string
import re


def remove_punctuation(text):
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
