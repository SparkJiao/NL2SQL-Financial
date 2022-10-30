import json
import argparse

empty_struct = {
    "from": {
        "table_units": [],
        "conds": []
    },
    "select": [],
    "where": [],
    "groupBy": [],
    "having": [],
    "orderBy": [],
    "limit": None
}


def extract_and_discriminate(structure):
    sub_struct = {
        "from": structure["from"],
        "select": structure["select"],
        "where": structure["where"],
        "groupBy": structure["groupBy"],
        "having": structure["having"],
        "orderBy": structure["orderBy"],
        "limit": structure["limit"],
    }

    return sub_struct == empty_struct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    data = json.load(open(args.input_file))

    outputs = []
    for item in data:
        if extract_and_discriminate(item):
            continue

        outputs.append(item)
