import json
import os

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.tokenization_utils import PaddingStrategy

from data.collators import DictTensorDataset
from data.data_utils import tokenizer_get_name
from general_util.logger import get_child_logger

logger = get_child_logger("RetrievalData")

db_vocab = {
    "ccks_stock": 0,
    "ccks_fund": 1,
    "ccks_macro": 2,
}


def table_name_ranking(file_path, tokenizer: PreTrainedTokenizer, db_info_path: str, max_seq_length: int = 256):
    # FIXME: Here is a problem that ``all_answer_id`` include only one positive candidates and the other positive ones are ignored.
    #   Besides, since the size of the negative candidate pool are small (size == 78), leading to repeat negative candidates during
    #   in-batch negative sampling.
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_seq_length}_retrieval_base_v1"
    cached_file_path = f"{file_path}_{file_suffix}"
    cached_file_path = cached_file_path.replace('*', '')
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        model_inputs, meta_data = torch.load(cached_file_path)
        return DictTensorDataset(model_inputs, meta_data)

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    db_info = json.load(open(db_info_path, 'r'))
    # all_table_names = []
    all_table_names_en2cn = {}
    for db in db_info:
        db_table_names = db["table_name"]
        # all_table_names.extend(db_table_names)
        all_table_names_en2cn.update({tab_en: tab_cn for tab_en, tab_cn in db_table_names})
    table_vocab = {tab_name: idx for idx, tab_name in enumerate(list(all_table_names_en2cn.values()))}

    all_query = []
    all_answers = []
    all_answer_id = []

    out_of_db = 0
    for item in tqdm(data, desc="Processing data", total=len(data)):
        question = item["question"]
        for tab_name in item["from"]["table_units"]:
            if tab_name not in all_table_names_en2cn:
                out_of_db += 1
                continue
            all_query.append(question)
            all_answers.append(all_table_names_en2cn[tab_name])
            all_answer_id.append(table_vocab[all_table_names_en2cn[tab_name]])

    # model_inputs = tokenizer(all_query, all_answers, max_length=max_seq_length, padding=PaddingStrategy.LONGEST, truncation=True)
    all_answer_id = torch.tensor(all_answer_id, dtype=torch.long)
    # model_inputs["answer_id"] = all_answer_id
    model_inputs = {}

    _que_inputs = tokenizer(all_query, max_length=max_seq_length, padding=PaddingStrategy.LONGEST, truncation=True, return_tensors="pt")
    model_inputs.update({f"que_{k}": v for k, v in _que_inputs.items()})

    _ctx_inputs = tokenizer(all_answers, max_length=max_seq_length, padding=PaddingStrategy.LONGEST, truncation=True, return_tensors="pt")
    model_inputs.update({f"ctx_{k}": v for k, v in _ctx_inputs.items()})

    model_inputs["answer_id"] = all_answer_id

    logger.info(f"Out of data base: {out_of_db}")

    return DictTensorDataset(model_inputs)


def table_name_ranking_all_neg(file_path, tokenizer: PreTrainedTokenizer, db_info_path: str, max_seq_length: int = 256):
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_seq_length}_retrieval_all_neg_v1"
    cached_file_path = f"{file_path}_{file_suffix}"
    cached_file_path = cached_file_path.replace('*', '')
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        model_inputs, meta_data = torch.load(cached_file_path)
        return DictTensorDataset(model_inputs, meta_data)

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    db_info = json.load(open(db_info_path, 'r'))
    all_table_names_en2cn = {}
    for db in db_info:
        db_table_names = db["table_name"]
        all_table_names_en2cn.update({tab_en: tab_cn for tab_en, tab_cn in db_table_names})
    table_vocab = {tab_name: idx for idx, tab_name in enumerate(list(all_table_names_en2cn.values()))}

    all_query = []
    all_answer_id = []
    all_mask_id = []

    out_of_db = 0
    for item in tqdm(data, desc="Processing data", total=len(data)):
        question = item["question"]
        answer_ids = []
        for tab_name in item["from"]["table_units"]:
            if tab_name not in all_table_names_en2cn:
                out_of_db += 1
                continue
            answer_ids.append(table_vocab[all_table_names_en2cn[tab_name]])
        if len(answer_ids) > 0:
            for idx in answer_ids:
                all_query.append(question)
                all_answer_id.append(idx)
                _tmp = [0] * len(table_vocab)
                for x in answer_ids:
                    if x != idx:
                        _tmp[x] = 1
                all_mask_id.append(_tmp)

    model_inputs = tokenizer(all_query, max_length=max_seq_length, padding=PaddingStrategy.LONGEST, truncation=True, return_tensors="pt")

    model_inputs["answer_id"] = torch.tensor(all_answer_id, dtype=torch.long)
    model_inputs["answer_mask"] = torch.tensor(all_mask_id, dtype=torch.long)  # [data_num, candidate_num]

    logger.info(f"Out of data base: {out_of_db}")

    if len(all_answer_id) == 0:
        return DictTensorDataset(model_inputs)
    return DictTensorDataset(model_inputs)


def table_name_ranking_eval(file_path, tokenizer: PreTrainedTokenizer, db_info_path: str, max_seq_length: int = 256):
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_seq_length}_retrieval_base_eval_v1"
    cached_file_path = f"{file_path}_{file_suffix}"
    cached_file_path = cached_file_path.replace('*', '')
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        model_inputs, meta_data = torch.load(cached_file_path)
        return DictTensorDataset(model_inputs, meta_data)

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    db_info = json.load(open(db_info_path, 'r'))
    # all_table_names = []
    all_table_names_en2cn = {}
    for db in db_info:
        db_table_names = db["table_name"]
        # all_table_names.extend(db_table_names)
        all_table_names_en2cn.update({tab_en: tab_cn for tab_en, tab_cn in db_table_names})
    table_vocab = {tab_name: idx for idx, tab_name in enumerate(list(all_table_names_en2cn.values()))}

    all_query = []
    all_answer_id = []

    out_of_db = 0
    for item in tqdm(data, desc="Processing data", total=len(data)):
        question = item["question"]
        answer_ids = []
        if "from" in item:
            for tab_name in item["from"]["table_units"]:
                if tab_name not in all_table_names_en2cn:
                    out_of_db += 1
                    continue
                answer_ids.append(table_vocab[all_table_names_en2cn[tab_name]])
            # if len(answer_ids) == 0:
            #     continue
            all_answer_id.append({"answer_id": answer_ids})

        all_query.append(question)

    model_inputs = tokenizer(all_query, max_length=max_seq_length, padding=PaddingStrategy.LONGEST, truncation=True, return_tensors="pt")

    logger.info(f"Out of data base: {out_of_db}")

    if len(all_answer_id) == 0:
        return DictTensorDataset(model_inputs)
    return DictTensorDataset(model_inputs, meta_data=all_answer_id)


def get_candidate_inputs(db_info_path: str, tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    db_info = json.load(open(db_info_path, 'r'))
    all_table_names = []
    for db in db_info:
        db_table_names = db["table_name"]
        all_table_names.extend([tab_name[1] for tab_name in db_table_names])  # chinese

    candidate_inputs = tokenizer(all_table_names, padding=PaddingStrategy.LONGEST, return_tensors="pt")

    # logger.info(candidate_inputs.__class__.__name__)

    return candidate_inputs
