import xml.etree.ElementTree as ET
from answer_query import answer_query


def test(path):
    tree = ET.parse(path)
    root = tree.getroot()

    queries = root.findall("./QUERY")
    out_file_name = "ranked_query_docs.txt"

    for query in queries:
        query_number = query.findall("./QueryNumber")[0].text.strip()
        query_text = query.findall("./QueryText")[0].text
        num_results_expected = int(query.findall("./Results")[0].text.strip())

        items = query.findall("./Records/Item")
        documents_expected = []

        for item in items:
            documents_expected.append(item.text)

        answer_query("vsm_inverted_index.json",
                     query_text,
                     out_file_name)

        with open(out_file_name, "r") as our_output:
            our_documents = our_output.readlines()

        assert our_documents == documents_expected and len(our_documents) == num_results_expected, f"query_number = {query_number}"


if __name__ == "__main__":
    test("./cfc-xml/cfquery.xml")
