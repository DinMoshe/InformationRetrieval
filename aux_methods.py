from nltk.stem import PorterStemmer
# import string

# define punctuation
punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

stopwords_set = {'very', 'hadn', 'mustn', 'ours', 'off', 'just', 'who', 'having', 'yours', 'not', 'if', "that'll",
                 'as', 'at', 'during', 'm', 'her', "mightn't", 'they', 'be', 'what', 'to', 'my', 'an', 'yourself',
                 'only', "hadn't", 'own', 'between', 'more', 're', "hasn't", 'has', 'until', 'for', 'because', 'into',
                 'ain', 'our', 'ourselves', 'few', "she's", "won't", 'theirs', 'of', 'now', 'by', 'under', 'each',
                 'with', "don't", 'do', 'ma', 'couldn', 'most', "shan't", 'd', "weren't", 'no', "should've", 'your',
                 'he', 'other', 'once', 'about', 't', 'doesn', 'against', 'have', 'below', 'aren', 'was', 'then',
                 "aren't", 'did', 'll', 'mightn', 'there', 'can', "couldn't", 'me', 'nor', 'we', "mustn't", 'that',
                 'needn', 'how', 'should', 'been', 'this', 'on', "wasn't", 'those', 've', 'she', 'and', 'yourselves',
                 'being', 'which', "you'll", 'a', 'you', 'won', 'them', 'i', 'itself', 'out', 'didn', "it's", 'again',
                 "haven't", "needn't", 'had', 'doing', 'or', 'any', 'both', 'why', 'too', 's', 'o', 'y', 'it',
                 'such', 'haven', "shouldn't", 'myself', 'herself', 'but', 'wasn', 'whom', 'after', 'from', 'over',
                 'all', 'am', 'weren', 'don', 'isn', 'were', 'same', 'shan', 'the', 'in', "isn't", 'where', 'above',
                 'up', 'here', 'through', 'themselves', 'shouldn', 'hers', 'hasn', 'himself', 'before', "you'd",
                 "wouldn't", 'these', 'so', 'its', 'his', 'is', 'are', 'some', 'wouldn', 'further', "you're", "you've",
                 'does', 'will', "didn't", 'down', 'him', 'while', 'when', 'their', "doesn't", 'than', '+'}


def remove_punctuation(s):
    for char in punctuation:
        s = s.replace(char, "")
    # s.translate(str.maketrans('', '', string.punctuation))
    return s


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
