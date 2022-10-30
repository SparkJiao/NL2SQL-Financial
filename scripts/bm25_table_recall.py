import argparse
import json
import sys
from typing import List, Dict

import jieba
from tqdm import tqdm

# pwd = os.getcwd()
# f_pwd = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")

sys.path.append('../')

from data.bm25 import BM25Model


def process(db_info: List, data: List[Dict], top_k_table_names: int = 5, in_db: bool = False):
    table_names_cn2en = {}
    db_bm25_model = {}

    for db in db_info:
        db_table_names = db["table_name"]
        db_table_cn2en = {item[1]: item[0] for item in db_table_names}
        table_names_cn2en.update(db_table_cn2en)
        db_bm25_model[db["db_name"]] = BM25Model([list(jieba.cut(k[1])) for k in db_table_names])

    print(len(table_names_cn2en))

    all_names_cn = list(table_names_cn2en.keys())

    names_list = [list(jieba.cut(k)) for k in all_names_cn]
    bm25_model = BM25Model(names_list)

    all_q_recall = []
    for item in tqdm(data, desc="Processing data", total=len(data)):
        # all_inputs.append(item["question"])
        db_name = item["db_name"]
        question = item["question"]

        ques_words = list(jieba.cut(question))
        if not in_db:
            bm25_scores = [(idx, _score) for idx, _score in enumerate(bm25_model.get_documents_score(ques_words))]
            sorted_bm25_scores = sorted(bm25_scores, key=lambda x: x[1], reverse=True)
            top_k_table_names_en = [table_names_cn2en[all_names_cn[idx]] for idx, _ in sorted_bm25_scores[:top_k_table_names]]
        else:
            bm25_scores = [(idx, _score) for idx, _score in enumerate(db_bm25_model[db_name].get_documents_score(ques_words))]
            sorted_bm25_scores = sorted(bm25_scores, key=lambda x: x[1], reverse=True)
            top_k_table_names_en = [table_names_cn2en[all_names_cn[idx]] for idx, _ in sorted_bm25_scores[:top_k_table_names]]

        target_table_names = item["from"]["table_units"]

        _recall = len(set(top_k_table_names_en) & set(target_table_names)) * 1.0 / len(target_table_names)
        all_q_recall.append(_recall)

    recall = sum(all_q_recall) / len(all_q_recall)
    return recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_info', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--in_db', default=False, action='store_true')
    args = parser.parse_args()

    db_info = json.load(open(args.db_info, 'r'))
    data = json.load(open(args.data, 'r'))
    recall = process(db_info, data, top_k_table_names=args.top_k, in_db=args.in_db)

    print("Recall: {}".format(recall))


if __name__ == '__main__':
    main()
