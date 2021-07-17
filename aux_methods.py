import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# define punctuation
punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))
stopwords_set.add("+")


def remove_punctuation(string):
    for char in punctuation:
        string = string.replace(char, "")
    return string


def tokenize_and_preprocess(text):
    """
    :param text: text to tokenize and preprocess
    :return: list of tokens.
    """
    list_words = text.split()

    # remove punctuation
    for i in range(len(list_words)):
        list_words[i] = remove_punctuation(list_words[i]).lower()

    # to use stemming:
    # 1. from nltk.stem import PorterStemmer
    # 2. ps = PorterStemmer()
    # 3. replace word with ps.stem(word)
    ps = PorterStemmer()
    tokens_without_stopwords = [ps.stem(word) for word in list_words
                                if word not in stopwords_set and word != "" and not word.isnumeric()]

    return tokens_without_stopwords
