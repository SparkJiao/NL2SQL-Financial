import json
from typing import List

from transformers import PreTrainedTokenizer

tokenizer: PreTrainedTokenizer
entity_name = None
entity_enum = None


def entity_labeling_init(_tokenizer: PreTrainedTokenizer, _ent_name, _ent_enum):
    global tokenizer
    global entity_name
    global entity_enum

    tokenizer = _tokenizer
    entity_name = _ent_name
    entity_enum = _ent_enum


def load_kb_entity_enum(file: str):
    kb = json.load(open(file))

    entity_name_set = set()
    entity_enum_set = set()
    for entity in kb:
        for ent_attr in entity["entityAttributes"]:
            attr_name = [ent_attr["attrName"]] + ent_attr["attrAlias"]
            entity_name_set.update(set(attr_name))

            attr_enum = ent_attr["attrEnum"]
            entity_enum_set.update(attr_enum)

    return entity_name_set, entity_enum_set


def annotate_span(tokens: List[str], span: str, _tokenizer: PreTrainedTokenizer, labels: List[int]):
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            tmp = _tokenizer.convert_tokens_to_string(tokens[i: j + 1])
            if tmp == span:
                # return True, i, j + 1
                labels[i: j + 1] = [1] * (j - i + 1)
                return True
    # return False, -1, -1
    return False


def annotate_entity(input_ids: List[int]):
    orig_len = len(input_ids)

    s_index = 0
    if len(tokenizer.tokenize("<input>")) == 1:
        s_index = input_ids.index(tokenizer.convert_tokens_to_ids(["<input>"])[0])
        if s_index == -1:
            # print("===========")
            s_index = 0
    e_index = input_ids.index(tokenizer.convert_tokens_to_ids(["<table>"])[0])

    # print(s_index, e_index)

    input_ids = input_ids[s_index: e_index]
    # print(len(input_ids))
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # print(tokens)
    question = tokenizer.convert_tokens_to_string(tokens)
    # print(question)
    name_labels = [0] * len(tokens)
    enum_labels = [0] * len(tokens)
    name_cnt = 0
    enum_cnt = 0

    for name in entity_name:
        if name not in question:
            continue
        flag1 = annotate_span(tokens, name, tokenizer, name_labels)
        if flag1:
            name_cnt += 1

    for enum in entity_enum:
        if enum not in question:
            continue
        flag2 = annotate_span(tokens, enum, tokenizer, enum_labels)
        if flag2:
            enum_cnt += 1

    name_labels = [0] * s_index + name_labels + [0] * (orig_len - e_index)
    enum_labels = [0] * s_index + enum_labels + [0] * (orig_len - e_index)
    assert len(name_labels) == len(enum_labels) == orig_len, (len(name_labels), len(enum_labels), orig_len)

    return name_labels, enum_labels, name_cnt, enum_cnt
