import json
import argparse
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    data = json.load(open(args.train_file))
    index_ls = list(range(len(data)))
    valid_id_ls = set(random.sample(index_ls, args.valid_num))

    sub_train = []
    sub_dev = []
    for idx, item in enumerate(data):
        if idx in valid_id_ls:
            sub_dev.append(item)
        else:
            sub_train.append(item)

    json.dump(sub_train, open(args.train_file.replace(".json", f"_sub_{len(sub_train)}_{args.seed}.json"), "w"), indent=2)
    json.dump(sub_dev, open(args.train_file.replace(".json", f"_sub_{len(sub_dev)}_{args.seed}.json"), "w"), indent=2)


if __name__ == '__main__':
    main()
