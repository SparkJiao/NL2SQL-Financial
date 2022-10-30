import argparse
import copy
import json


def extract_structure(item):
    return {
        "db_name": item["db_name"],
        "from": item["from"],
        "select": item["select"],
        "where": item["where"],
        "groupBy": item["groupBy"],
        "having": item["having"],
        "orderBy": item["orderBy"],
        "limit": item["limit"]
    }


def extract_wrong_predictions(predictions, data):
    # id2pred = {pred["q_id"]: pred for pred in predictions}
    id2data = {item["q_id"]: item for item in data}

    wrong_predictions = []
    correct = 0
    for pred in predictions:
        item = id2data[pred["q_id"]]
        if extract_structure(item) != extract_structure(pred):
            tmp = copy.deepcopy(pred)
            tmp["data"] = item
            wrong_predictions.append(tmp)
        else:
            correct += 1

    print("Accuracy: {}".format(correct / len(predictions)))
    return wrong_predictions


def main():
    parser = argparse.ArgumentParser(description='Error analysis')
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    args = parser.parse_args()

    data = json.load(open(args.data_file))
    predictions = json.load(open(args.predictions))
    wrong_predictions = extract_wrong_predictions(predictions, data)

    output_file = args.predictions.replace(".json", ".errors.json")
    json.dump(wrong_predictions, open(output_file, "w"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
