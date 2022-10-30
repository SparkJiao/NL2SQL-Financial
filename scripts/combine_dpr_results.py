import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_results_file", type=str, required=True)
    parser.add_argument("--dev_results_file", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--new_train_file", type=str)
    parser.add_argument("--new_dev_file", type=str)
    args = parser.parse_args()

    train_dpr_results = json.load(open(args.train_results_file, "r"))
    train_data = json.load(open(args.train_file, "r"))
    dev_dpr_results = json.load(open(args.dev_results_file, "r"))
    dev_data = json.load(open(args.dev_file, "r"))

    combine_data = {}
    for data, dpr_results in zip(train_data, train_dpr_results):
        combine_data[data["q_id"]] = {
            "data": data,
            "dpr_res": dpr_results
        }
    for data, dpr_results in zip(dev_data, dev_dpr_results):
        combine_data[data["q_id"]] = {
            "data": data,
            "dpr_res": dpr_results
        }

    new_train = json.load(open(args.new_train_file, "r"))
    new_dev = json.load(open(args.new_dev_file, "r"))

    new_train_dpr = []
    new_dev_dpr = []
    for item in new_train:
        item_dpr = combine_data[item["q_id"]]["dpr_res"]
        new_train_dpr.append(item_dpr)
    for item in new_dev:
        item_dpr = combine_data[item["q_id"]]["dpr_res"]
        new_dev_dpr.append(item_dpr)

    json.dump(new_train_dpr, open(args.new_train_file.replace(".json", "_dpr.json"), "w"))
    json.dump(new_dev_dpr, open(args.new_dev_file.replace(".json", "_dpr.json"), "w"))


if __name__ == '__main__':
    main()
