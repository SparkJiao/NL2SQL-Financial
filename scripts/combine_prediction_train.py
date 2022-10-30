import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, required=True)
parser.add_argument("--prediction_file", type=str, required=True)
parser.add_argument("--output_file", type=str, default=None)

args = parser.parse_args()

predictions = json.load(open(args.prediction_file))
train = json.load(open(args.train_file))

train.extend(predictions)

if args.output_file is None:
    json.dump(train, open(args.prediction_file.replace(".json", ".train_ext.json"), "w"), indent=2)
else:
    json.dump(train, open(args.output_file, "w"), indent=2)
print("Done")

