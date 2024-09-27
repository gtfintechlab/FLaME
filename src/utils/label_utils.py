def encode(label_word):

    if label_word == "positive":

        return 0

    elif label_word == "negative":

        return 1

    elif label_word == "neutral":

        return 2

    else:

        return -1
