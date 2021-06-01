from aux_methods import tokenize_and_preprocess
import json
from collections import Counter
import numpy as np


def answer_query(index_path, query, out_path):
    """
    :param out_path: the path+filename where the results will be saved
    :param index_path: the path where the index is stored
    :param query: user query to answer
    :return: a list of relevant documents sorted in descending order by their cosine similarity
    to the query.
    """

    tokens_without_stopwords = tokenize_and_preprocess(query)

    counter_tokens = Counter(tokens_without_stopwords)

    max_occurrences = max(counter_tokens.values())

    for token in counter_tokens:
        counter_tokens[token] /= max_occurrences

    with open(index_path, 'r') as file_json:
        json_index_data = json.load(file_json)

    query_length = 0
    cosine_similarity_dict = dict()  # map between doc_id and its cosine similarity with the query

    for token in tokens_without_stopwords:
        tf_in_query = counter_tokens[token]
        idf_token = float(json_index_data["idf"].get(token, 0))  # idf = 0 if token is not in the corpus
        token_weight = tf_in_query * idf_token
        query_length += token_weight ** 2
        tf_map = json_index_data["tf"].get(token, dict())

        for doc_id, tf in tf_map.items():
            # calculate inner product between the query and doc_id
            if doc_id not in cosine_similarity_dict:
                cosine_similarity_dict[doc_id] = 0.0

            cosine_similarity_dict[doc_id] += token_weight * idf_token * tf

    query_length = np.sqrt(query_length)

    # normalize by the lengths of the query and the document
    for doc_id in cosine_similarity_dict:
        cosine_similarity_dict[doc_id] /= query_length * float(json_index_data["documents_length"][doc_id])

    with open(out_path, "w") as out:
        out.writelines("\n".join(sorted(cosine_similarity_dict, key=cosine_similarity_dict.get)))


# if __name__ == "__main__":
#     answer_query("vsm_inverted_index.json",
#                  "Is salt (sodium and/or chloride) transport/permeability abnormal in CF?",
#                  "ranked_query_docs.txt")
