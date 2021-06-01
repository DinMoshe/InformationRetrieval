from aux_methods import tokenize_and_preprocess
import xml.etree.ElementTree as ET
import os
import numpy as np
import json


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

        max_occurrences_dict[document_id] = max(max_occurrences_dict.get(document_id, 0),
                                                self.tf_map[document_id])


class InvertedIndex:
    index_term_hash: dict  # a map between an index term to a TokenInfo object
    num_documents: int  # the number of documents in the corpus
    corpus_directory: str  # the directory in which the corpus is
    filenames: list  # list of filenames to go through in corpus_directory
    idf_scores: dict  # map between a token and its idf score. It is calculated after creating the inverted index.
    documents_length: dict  # a map between a document id and its length as a vector.
    max_occurrences: dict  # a map between a document id and its maximal number of occurrences of a token.
    json_filename: str  # the name of the json file to load the index to

    def __init__(self, corpus_directory: str, filenames: list, json_filename: str):
        self.index_term_hash = dict()
        self.idf_scores = dict()
        self.documents_length = dict()
        self.max_occurrences = dict()
        self.corpus_directory = corpus_directory
        self.filenames = filenames
        self.num_documents = 0
        self.json_filename = json_filename

    def process_text(self, text, doc_id):
        """
        :param text: text to process
        :param doc_id: document id to whom the text belongs.
        :return:
        """

        tokens_without_stopwords = tokenize_and_preprocess(text)

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
        :return: This method creates and saves the inverted index in a json file called self.json_filename
        """

        for filename in self.filenames:
            # if self.num_documents == 5:
            #     break

            tree = ET.parse(os.path.join(self.corpus_directory, filename))
            root = tree.getroot()

            documents = root.findall("./RECORD")
            for doc in documents:
                self.num_documents += 1
                doc_id = doc.findall("./RECORDNUM")[0].text.strip()
                doc_id = str(int(doc_id))  # remove preceding zeros
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
                    self.process_text(text, doc_id)

                # if self.num_documents == 5:
                #     break

        self.compute_idf()
        self.compute_documents_length()

        # now we have all the data we want

        self.save_json()

    def compute_idf(self):
        """
            To be called only after build_inverted_index
        :return: This method calculates the idf scores for every token in self.index_term_hash dictionary.
        """
        for token in self.index_term_hash.keys():
            df_score = self.index_term_hash[token].df_score
            self.idf_scores[token] = np.log2(self.num_documents / df_score)

    def compute_documents_length(self):
        """
            To be called only after compute_idf
        :return: This method calculates the norm (length) of each document in the corpus. It also normalizes the
            tf scores for each token and a document by the maximal number of occurrences of a token in this document.
        """
        for token in self.index_term_hash.keys():
            idf_score = self.idf_scores[token]
            token_info = self.index_term_hash[token]
            tf_map = token_info.tf_map
            for doc_id in tf_map.keys():
                tf_normalized = tf_map[doc_id] / self.max_occurrences[doc_id]
                tf_map[doc_id] = tf_normalized
                self.documents_length[doc_id] += (idf_score * tf_normalized) ** 2

        for doc_id in self.documents_length.keys():
            self.documents_length[doc_id] = np.sqrt(self.documents_length[doc_id])

    def save_json(self):
        dict_to_save = {"tf": dict(),
                        "idf": self.idf_scores,
                        "documents_length": self.documents_length,
                        "num_documents": self.num_documents}

        for token in self.index_term_hash.keys():
            token_info = self.index_term_hash[token]
            dict_to_save["tf"][token] = token_info.tf_map  # reminder: token_info.tf_map is a dictionary

        with open(self.json_filename, 'w') as json_file:
            json.dump(dict_to_save, json_file, sort_keys=True, indent=4)


# if __name__ == "__main__":
#     filenames = [f"cf{num}.xml" for num in range(74, 80)]
#     corpus_directory = "cfc-xml"
#     index = InvertedIndex(corpus_directory, filenames, "vsm_inverted_index.json")
#     index.build_inverted_index()
