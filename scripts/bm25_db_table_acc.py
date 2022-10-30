import argparse
import collections
import json
import sys
from typing import List, Dict

import jieba
from tqdm import tqdm

# pwd = os.getcwd()
# f_pwd = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")

sys.path.append('../')

from data.bm25 import BM25Model


def process(db_info: List, data: List[Dict], top_k: int = 3):
    db_document = []
    db_ids = []
    for item in tqdm(db_info, desc="Building BM25 model", total=len(db_info)):
        # db_item = item["db_id"]
        db_item = list(jieba.cut(item["db_id"]))
        tables = item["table_names"]
        table2col = collections.defaultdict(list)
        for col_id, col_name in item["column_names"]:
            if col_id == -1:
                continue
            table2col[tables[col_id]].append(col_name)
        for tab_name, col_name_ls in table2col.items():
            # db_item = " ".join([db_item, tab_name] + col_name_ls)
            db_item.extend(list(jieba.cut(tab_name)))
            for col in col_name_ls:
                db_item.extend(list(jieba.cut(col)))
        db_document.append(db_item)
        db_ids.append(item["db_id"])

    bm25_model = BM25Model(db_document)

    all_recall = []
    for item in tqdm(data, desc="Processing data", total=len(data)):
        db_id = item["db_id"]
        question = item["question"]

        ques_words = list(jieba.cut(question))
        bm25_scores = [(idx, _score) for idx, _score in enumerate(bm25_model.get_documents_score(ques_words))]
        sorted_bm25_scores = sorted(bm25_scores, key=lambda x: x[1], reverse=True)

        top_k_db = [db_ids[idx] for idx, _ in sorted_bm25_scores[:top_k]]

        _recall = len(set(top_k_db) & {db_id}) * 1.0
        all_recall.append(_recall)

    recall = sum(all_recall) / len(all_recall)
    return recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_info', type=str, required=True)  # DuSQL
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    db_info = json.load(open(args.db_info, 'r'))
    data = json.load(open(args.data, 'r'))
    recall = process(db_info, data, top_k=args.top_k)

    print("Recall: {}".format(recall))


if __name__ == '__main__':
    main()
