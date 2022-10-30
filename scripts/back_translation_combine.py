import copy
import json
import argparse
import jieba
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nl2sql_pred_file", type=str, required=True)
    parser.add_argument("--sql2nl_pred_file", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--bleu_threshold", type=float, default=0.0)  # 0.0 means keep all generated sql.
    parser.add_argument("--keep_nl", default=False, action="store_true")
    parser.add_argument("--keep_all_nl", default=False, action="store_true")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    nl2sql_predictions = json.load(open(args.nl2sql_pred_file, 'r'))
    sql2nl_predictions = json.load(open(args.sql2nl_pred_file, 'r'))

    pseudo_examples = []
    kept_num = 0
    all_kept_nl_num = 0
    for item1, item2 in tqdm(zip(nl2sql_predictions, sql2nl_predictions), total=len(nl2sql_predictions)):
        assert item1["q_id"] == item2["q_id"]

        bleu = sentence_bleu([list(jieba.cut(item1["question"]))], list(jieba.cut(item2["pred_question"])))

        if bleu > args.bleu_threshold:
            pseudo_examples.append(item1)
            kept_num += 1

            if args.keep_nl and item1["question"] != item2["pred_question"]:
                item = copy.deepcopy(item1)
                item["question"] = item2["pred_question"]
                pseudo_examples.append(item)

        if args.keep_all_nl and item1["question"] != item2["pred_question"]:
            item = copy.deepcopy(item1)
            item["question"] = item2["pred_question"]
            pseudo_examples.append(item)
            all_kept_nl_num += 1

    print(f"Kept data amount: {kept_num}")
    print(f"All kept nl data amount: {all_kept_nl_num}")

    train = json.load(open(args.train_file))
    train.extend(pseudo_examples)

    json.dump(train, open(args.output_file, "w"), indent=2)

    notation = {
        "nl2sql_pred": args.nl2sql_pred_file,
        "sql2nl_pred": args.sql2nl_pred_file,
        "train_file": args.train_file,
        "keep_nl": args.keep_nl,
        "bleu_threshold": args.bleu_threshold,
        "keep_all_nl": args.keep_all_nl,
    }
    json.dump(notation, open(args.output_file.replace(".json", ".note.json"), "w"), indent=2)


if __name__ == '__main__':
    main()
