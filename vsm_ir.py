import sys
import inverted_index
from answer_query import answer_query


def parse_cmd_line():
    if len(sys.argv) < 2:
        print("Not enough arguments")
        return

    action = sys.argv[1]
    if action == "create_index":
        if len(sys.argv) < 3:
            print("Not enough arguments")
            return
        corpus_directory = sys.argv[2]
        filenames = [f"cf{num}.xml" for num in range(74, 80)]
        index = inverted_index.InvertedIndex(corpus_directory, filenames, "vsm_inverted_index.json")
        index.build_inverted_index()

    elif action == "query":
        if len(sys.argv) < 4:
            print("Not enough arguments")
            return
        index_path = sys.argv[2]
        question = sys.argv[3]
        answer_query(index_path,
                     question,
                     "ranked_query_docs.txt")
    else:
        print("Illegal action")


if __name__ == "__main__":
    parse_cmd_line()
