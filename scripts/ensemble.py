import glob
import json
from argparse import ArgumentParser


def ensemble_score(data1, data2):
    outputs = []
    for item1, item2 in zip(data1, data2):
        assert item1["q_id"] == item2["q_id"]
        if item1["sequence_scores"] > item2["sequence_scores"]:
            outputs.append(item1)
        else:
            outputs.append(item2)
    return outputs


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file1", type=str, required=True)
    parser.add_argument("--input_file2", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    data1 = json.load(open(args.input_file1, 'r'))
    data2 = json.load(open(args.input_file2, 'r'))

    outputs = ensemble_score(data1, data2)
    json.dump(outputs, open(args.output_file, "w"), indent=2)


if __name__ == '__main__':
    main()
