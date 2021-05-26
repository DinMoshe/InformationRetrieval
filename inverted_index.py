import xml.etree.ElementTree as ET
import os
import numpy as np

filenames = [f"cf{num}.xml" for num in range(74, 80)]

# define punctuation
punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

with open("stopwords.txt", "r") as stop_file:
    stopwords = stop_file.readlines()


def remove_punctuation(string):
    for char in punctuation:
        string = string.replace(char, "")
    return string


class TokenInfo:
    token: str  # the token itself
    df_score: int  # the df score of the token associated with this object
    tf_map: dict  # a map between a document id to the number of occurrences of the token in this document

    def __init__(self, token: str):
        self.token = token
        self.tf_map = dict()
        self.df_score = 0

    def update_token_info(self, document_id, max_occurrences_dict):
        """
        :param max_occurrences_dict: to be received by InvertedIndex. A map between a document id and its maximal number
        of occurrences of a token.
        :param document_id: the document id for which to increase the number of occurrences for.
        :return: updates the df_score and tf_map
        """
        self.df_score += 1
        if document_id in self.tf_map:
            self.tf_map[document_id] += 1
        else:
            self.tf_map[document_id] = 1

        max_occurrences_dict[document_id] = max(max_occurrences_dict[document_id], self.tf_map[document_id])


class InvertedIndex:
    index_term_hash: dict  # a map between an index term to a TokenInfo object
    num_documents: int  # the number of documents in the corpus
    corpus_directory: str  # the directory in which the corpus is
    filenames: list  # list of filenames to go through in corpus_directory
    idf_scores: dict  # map between a token and its idf score. It is calculated after creating the inverted index.
    documents_length: dict  # a map between a document id and its length as a vector.
    max_occurrences: dict  # a map between a document id and its maximal number of occurrences of a token.

    def __init__(self, corpus_directory: str, filenames: list):
        self.index_term_hash = dict()
        self.idf_scores = dict()
        self.documents_length = dict()
        self.max_occurrences = dict()
        self.corpus_directory = corpus_directory
        self.filenames = filenames
        self.num_documents = 0

    def tokenize_and_preprocess(self, text, doc_id):
        """
        :param text: text to process
        :param doc_id: document id to whom the text belongs.
        :return:
        """
        list_words = text.split()

        # remove punctuation
        for i in range(len(list_words)):
            list_words[i] = remove_punctuation(list_words[i])

        tokens_without_stopwords = [word for word in list_words if word not in stopwords]

        # we update the inverted index (update self.index_term_hash)
        for token in tokens_without_stopwords:
            if token in self.index_term_hash:
                token_info = self.index_term_hash[token]
                token_info.update_token_info(document_id=doc_id, max_occurrences_dict=self.max_occurrences)
            else:
                token_info = TokenInfo(token)
                token_info.update_token_info(document_id=doc_id, max_occurrences_dict=self.max_occurrences)
                self.index_term_hash[token] = token_info

    def build_inverted_index(self):
        """
        :return: This method creates and saves the inverted index in a json file called "vsm_inverted_index.json"
        """

        for filename in self.filenames:
            tree = ET.parse(os.path.join(self.corpus_directory, filename))
            root = tree.getroot()

            documents = root.findall("./RECORD")
            for doc in documents:
                self.num_documents += 1
                doc_id = doc.findall("./RECORDNUM")[0].text.strip()
                self.documents_length[doc_id] = 0  # initialize the document length to 0 for afterwards
                title = doc.findall("./TITLE")[0].text
                list_of_text = [title]
                abstract = doc.findall("./ABSTRACT")
                extract = doc.findall("./EXTRACT")

                if len(abstract) > 0:
                    # according to the dtd there is only one such element
                    list_of_text.append(abstract[0].text)

                if len(extract) > 0:
                    # according to the dtd there is only one such element
                    list_of_text.append(extract[0].text)

                for text in list_of_text:
                    self.tokenize_and_preprocess(text, doc_id)

    def compute_idf(self):
        """
            To be called only after build_inverted_index
        :return: It calculates the idf scores for every token in self.index_term_hash dictionary.
        """
        for token in self.index_term_hash.keys():
            df_score = self.index_term_hash[token].df_score
            self.idf_scores[token] = np.log2(self.num_documents / df_score)

    def compute_documents_length(self):
        """
            To be called only after compute_idf
        :return:
        """
        for token in self.index_term_hash.keys():
            idf_score = self.idf_scores[token]
            token_info = self.index_term_hash[token]
            tf_map = token_info.tf_map
            for doc_id in tf_map.keys():
                tf_normalized = tf_map[doc_id] / self.max_occurrences[doc_id]
                self.documents_length[doc_id] += (idf_score * tf_normalized) ** 2

        for doc_id in self.documents_length.keys():
            self.documents_length[doc_id] = np.sqrt(self.documents_length[doc_id])

# if __name__ == "__main__":
#     index = InvertedIndex()
#     index.build_inverted_index("./cfc-xml", ["cf74.xml"])
