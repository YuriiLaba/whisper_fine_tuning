import numpy as np


def levenshtein(str_a, str_b, debug=False) -> tuple:
    """
    :param str_a: the hypothesis string
    :param str_b: the reference string
    :param debug: print edit distance matrix step by step
    :return: levenshtein distance between strings

    Time and space complexity of this implementation is O(len(str_a) * len(str_b))
    """

    # add padding to handle cases where one or both of the input strings are empty
    edit_dist_matrix = np.zeros((len(str_a) + 1, len(str_b) + 1), dtype=int)

    edit_dist_matrix[:, 0] = np.arange(len(str_a) + 1)
    edit_dist_matrix[0, :] = np.arange(len(str_b) + 1)

    for i in range(1, len(str_a) + 1):
        for j in range(1, len(str_b) + 1):
            deletion = edit_dist_matrix[i-1, j]
            replacement = edit_dist_matrix[i-1, j-1]
            insertion = edit_dist_matrix[i, j-1]

            edit_dist_matrix[i, j] = min([deletion, replacement, insertion])

            if str_a[i-1] != str_b[j-1]:
                edit_dist_matrix[i, j] += 1

            if debug:
                print(f"{i} {j}\n{edit_dist_matrix}\n")

    return edit_dist_matrix, edit_dist_matrix[-1, -1]


def wer(str_a, str_b) -> float:
    """
    :param str_a: the hypothesis string
    :param str_b: the reference string
    :return: word error rate
    """
    distance_matrix, num_errors = levenshtein(str_a.split(), str_b.split())
    num_words = len(str_b.split())
    return num_errors / num_words


def cer(str_a, str_b) -> float:
    """
    :param str_a: the hypothesis string
    :param str_b: the reference string
    :return: word error rate
    """
    distance_matrix, num_errors = levenshtein(str_a, str_b)
    num_characters = len(str_b)
    return num_errors / num_characters
