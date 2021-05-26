import sys

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

    elif action == "query":
        if len(sys.argv) < 4:
            print("Not enough arguments")
            return
        index_path = sys.argv[2]
        question = sys.argv[3]
    else:
        print("Illegal action")


if __name__ == "__main__":
    parse_cmd_line()

