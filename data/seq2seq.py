import copy
import json
import os

import jieba
import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy
from tqdm import tqdm
from typing import Union, List, Dict, Any

from data.bm25 import BM25Model
from data.collators import DictTensorDataset
from data.data_utils import tokenizer_get_name, sql_query_filter
from general_util.logger import get_child_logger

logger = get_child_logger("Seq2SeqData")

db_vocab = {
    "ccks_stock": 0,
    "ccks_fund": 1,
    "ccks_macro": 2,
}


def read_data_baseline(file_path, tokenizer: PreTrainedTokenizer, max_input_length: int = 128, max_output_length: int = 512):
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_input_length}_{max_output_length}_seq2seq_base_v1"
    cached_file_path = f"{file_path}_{file_suffix}"
    cached_file_path = cached_file_path.replace('*', '')
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        model_inputs, meta_data = torch.load(cached_file_path)
        return DictTensorDataset(model_inputs, meta_data)

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    all_inputs = []
    all_seq_queries = []
    all_cls_labels = []
    meta_data = []

    for item in data:
        all_inputs.append(item["question"])

        meta = {
            "q_id": item["q_id"],
            "question": item["question"],
        }

        if "db_name" in item:
            all_cls_labels.append(db_vocab[item["db_name"]])
            meta["db_name"] = item["db_name"]

        if "sql_query" in item:
            all_seq_queries.append(item["sql_query"])
            meta["sql_query"] = item["sql_query"]

        if "from" in item:
            meta["sqlite"] = {
                "from": item["from"],
                "select": item["select"],
                "where": item["where"],
                "groupBy": item["groupBy"],
                "having": item["having"],
                "orderBy": item["orderBy"],
                "limit": item["limit"]
            }

        meta_data.append(meta)

    logger.info("Read {} data points".format(len(all_inputs)))

    model_inputs = tokenizer(all_inputs, padding=PaddingStrategy.LONGEST, truncation=True, return_tensors="pt", max_length=max_input_length)
    logger.info(f"Input length: {model_inputs['input_ids'].size(1)}")
    if len(all_seq_queries):
        with tokenizer.as_target_tokenizer():
            model_seq_queries = tokenizer(all_seq_queries, padding=PaddingStrategy.LONGEST, truncation=True,
                                          max_length=max_output_length, return_tensors="pt")
            model_inputs["labels"] = model_seq_queries["input_ids"]
            logger.info(f"Seq2Seq output length: {model_inputs['labels'].size(1)}")
    if len(all_cls_labels):
        model_inputs["cls_labels"] = torch.tensor(all_cls_labels, dtype=torch.long)

    logger.info(f"Saving to {cached_file_path}.")
    torch.save((model_inputs, meta_data), cached_file_path)

    return DictTensorDataset(model_inputs, meta_data)


def read_sql2nl_baseline(file_path, tokenizer: PreTrainedTokenizer, max_input_length: int = 128, max_output_length: int = 256):
    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    all_questions = []
    all_seq_queries = []
    meta_data = []

    for item in data:
        all_questions.append(item["question"])

        meta = {
            "q_id": item["q_id"],
            "question": item["question"],
        }

        if "db_name" in item:
            meta["db_name"] = item["db_name"]

        if "sql_query" in item:
            all_seq_queries.append(item["sql_query"])
            meta["sql_query"] = item["sql_query"]

        meta_data.append(meta)

    logger.info("Read {} data points".format(len(all_questions)))

    model_inputs = tokenizer(all_seq_queries, padding=PaddingStrategy.LONGEST, truncation=True, return_tensors="pt",
                             max_length=max_input_length)
    logger.info(f"Input length: {model_inputs['input_ids'].size(1)}")

    with tokenizer.as_target_tokenizer():
        model_questions = tokenizer(all_questions, padding=PaddingStrategy.LONGEST, truncation=True,
                                    max_length=max_output_length, return_tensors="pt")
        model_inputs["labels"] = model_questions["input_ids"]
        logger.info(f"Seq2Seq output length: {model_inputs['labels'].size(1)}")

    return DictTensorDataset(model_inputs, meta_data)


def read_data_with_table_names(file_path, tokenizer: PreTrainedTokenizer, db_info_path: str, top_k_table_names: int = 5,
                               max_input_length: int = 128, max_output_length: int = 512):
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_input_length}_{max_output_length}_seq2seq_table_cn_top{top_k_table_names}_v1"
    cached_file_path = f"{file_path}_{file_suffix}"
    cached_file_path = cached_file_path.replace('*', '')
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        model_inputs, meta_data = torch.load(cached_file_path)
        return DictTensorDataset(model_inputs, meta_data)

    db_info = json.load(open(db_info_path, 'r'))

    table_names_cn2en = {}

    for db in db_info:
        db_table_names = db["table_name"]
        db_table_cn2en = {item[1]: item[0] for item in db_table_names}
        table_names_cn2en.update(db_table_cn2en)

    all_names_cn = list(table_names_cn2en.keys())

    names_list = [list(jieba.cut(k)) for k in all_names_cn]
    bm25_model = BM25Model(names_list)

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    all_inputs = []
    all_seq_queries = []
    all_cls_labels = []
    meta_data = []

    for item in tqdm(data, desc="Processing data", total=len(data)):
        # all_inputs.append(item["question"])
        question = item["question"]

        ques_words = list(jieba.cut(question))
        bm25_scores = [(idx, _score) for idx, _score in enumerate(bm25_model.get_documents_score(ques_words))]
        sorted_bm25_scores = sorted(bm25_scores, key=lambda x: x[1], reverse=True)
        top_k_table_names_en = [table_names_cn2en[all_names_cn[idx]] for idx, _ in sorted_bm25_scores[:top_k_table_names]]

        _input = " ".join([question, "<tables>"] + top_k_table_names_en)
        all_inputs.append(_input)

        meta = {
            "q_id": item["q_id"],
            "question": item["question"],
            "top_k_table_name": [
                [table_names_cn2en[all_names_cn[idx]], all_names_cn[idx]] for idx, _ in sorted_bm25_scores[:top_k_table_names]
            ]
        }

        if "db_name" in item:
            all_cls_labels.append(db_vocab[item["db_name"]])
            meta["db_name"] = item["db_name"]

        if "sql_query" in item:
            all_seq_queries.append(item["sql_query"])
            meta["sql_query"] = item["sql_query"]

        if "from" in item:
            meta["sqlite"] = {
                "from": item["from"],
                "select": item["select"],
                "where": item["where"],
                "groupBy": item["groupBy"],
                "having": item["having"],
                "orderBy": item["orderBy"],
                "limit": item["limit"]
            }

        meta_data.append(meta)

    logger.info("Read {} data points".format(len(all_inputs)))

    model_inputs = tokenizer(all_inputs, padding=PaddingStrategy.LONGEST, truncation=True, return_tensors="pt", max_length=max_input_length)
    logger.info(f"Input length: {model_inputs['input_ids'].size(1)}")
    if len(all_seq_queries):
        with tokenizer.as_target_tokenizer():
            model_seq_queries = tokenizer(all_seq_queries, padding=PaddingStrategy.LONGEST, truncation=True,
                                          max_length=max_output_length, return_tensors="pt")
            model_inputs["labels"] = model_seq_queries["input_ids"]
            logger.info(f"Seq2Seq output length: {model_inputs['labels'].size(1)}")
    if len(all_cls_labels):
        model_inputs["cls_labels"] = torch.tensor(all_cls_labels, dtype=torch.long)

    logger.info(f"Saving to {cached_file_path}.")
    torch.save((model_inputs, meta_data), cached_file_path)

    return DictTensorDataset(model_inputs, meta_data)


def read_data_with_table_names_dpr(file_path, tokenizer: PreTrainedTokenizer, db_info_path: str, dpr_results_path: str,
                                   top_k_table_names: int = 5, add_cn: bool = False,
                                   max_input_length: int = 128, max_output_length: int = 512):
    db_info = json.load(open(db_info_path, 'r'))

    all_table_names_en = []
    all_table_names_cn = []
    for db in db_info:
        db_table_names = db["table_name"]
        all_table_names_en.extend([tab_name[0] for tab_name in db_table_names])
        all_table_names_cn.extend([tab_name[1] for tab_name in db_table_names])

    dpr_results = json.load(open(dpr_results_path, 'r'))

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    all_inputs = []
    all_seq_queries = []
    all_cls_labels = []
    meta_data = []

    for item_id, item in enumerate(tqdm(data, desc="Processing data", total=len(data))):
        question = item["question"]

        top_k_table_names_en = [all_table_names_en[idx] for idx in dpr_results[item_id][:top_k_table_names]]
        top_k_table_names_cn = [all_table_names_cn[idx] for idx in dpr_results[item_id][:top_k_table_names]]

        if not add_cn:
            _input = " ".join([question, "<tables>"] + top_k_table_names_en)
        else:
            _input = question
            for idx, (en_tab, cn_tab) in enumerate(zip(top_k_table_names_en, top_k_table_names_cn)):
                _input = _input + f"<table{idx}>" + en_tab + " " + cn_tab
        all_inputs.append(_input)

        meta = {
            "q_id": item["q_id"],
            "question": item["question"],
            "top_k_table_name": [
                [name_cn, name_en] for name_cn, name_en in zip(top_k_table_names_cn, top_k_table_names_en)
            ]
        }

        if "db_name" in item:
            all_cls_labels.append(db_vocab[item["db_name"]])
            meta["db_name"] = item["db_name"]

        if "sql_query" in item:
            all_seq_queries.append(item["sql_query"])
            meta["sql_query"] = item["sql_query"]

        if "from" in item:
            meta["sqlite"] = {
                "from": item["from"],
                "select": item["select"],
                "where": item["where"],
                "groupBy": item["groupBy"],
                "having": item["having"],
                "orderBy": item["orderBy"],
                "limit": item["limit"]
            }

        meta_data.append(meta)

    logger.info("Read {} data points".format(len(all_inputs)))

    model_inputs = tokenizer(all_inputs, padding=PaddingStrategy.LONGEST, truncation=True, return_tensors="pt", max_length=max_input_length)
    logger.info(f"Input length: {model_inputs['input_ids'].size(1)}")
    if len(all_seq_queries):
        with tokenizer.as_target_tokenizer():
            model_seq_queries = tokenizer(all_seq_queries, padding=PaddingStrategy.LONGEST, truncation=True,
                                          max_length=max_output_length, return_tensors="pt")
            model_inputs["labels"] = model_seq_queries["input_ids"]
            logger.info(f"Seq2Seq output length: {model_inputs['labels'].size(1)}")
    if len(all_cls_labels):
        model_inputs["cls_labels"] = torch.tensor(all_cls_labels, dtype=torch.long)

    # logger.info(f"Saving to {cached_file_path}.")
    # torch.save((model_inputs, meta_data), cached_file_path)

    return DictTensorDataset(model_inputs, meta_data)


def read_data_with_table_column_names_simplify(file_path, tokenizer: PreTrainedTokenizer, db_info_path: str, top_k_table_names: int = 5,
                                               top_k_col: Union[str, int] = 5, max_input_length: int = 128, max_output_length: int = 512):
    tokenizer_name = tokenizer_get_name(tokenizer)

    file_suffix = f"{tokenizer_name}_{max_input_length}_{max_output_length}_seq2seq_tab_cn_top{top_k_table_names}_col_en_top{top_k_col}_v1"
    cached_file_path = f"{file_path}_{file_suffix}"
    cached_file_path = cached_file_path.replace('*', '')
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}.")
        model_inputs, meta_data = torch.load(cached_file_path)
        return DictTensorDataset(model_inputs, meta_data)

    db_info = json.load(open(db_info_path, 'r'))

    table_names_cn2en = {}
    table_names_en2col = {}

    for db in db_info:
        db_table_names = db["table_name"]
        db_table_cn2en = {item[1]: item[0] for item in db_table_names}
        table_names_cn2en.update(db_table_cn2en)
        for tab in db["column_info"]:
            tab_en_name = tab["table"]
            tab_col_en = tab["columns"][2:]  # 忽略 `*` 和 `id` 两列
            table_names_en2col[tab_en_name] = tab_col_en

    all_names_cn = list(table_names_cn2en.keys())

    names_list = [list(jieba.cut(k)) for k in all_names_cn]
    bm25_model = BM25Model(names_list)

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    all_inputs = []
    all_seq_queries = []
    all_cls_labels = []
    meta_data = []

    for item in tqdm(data, desc="Processing data", total=len(data)):
        # all_inputs.append(item["question"])
        question = item["question"]

        ques_words = list(jieba.cut(question))
        bm25_scores = [(idx, _score) for idx, _score in enumerate(bm25_model.get_documents_score(ques_words))]
        sorted_bm25_scores = sorted(bm25_scores, key=lambda x: x[1], reverse=True)
        top_k_table_names_en = [table_names_cn2en[all_names_cn[idx]] for idx, _ in sorted_bm25_scores[:top_k_table_names]]

        # _input = " ".join([question, "<tables>"] + top_k_table_names_en)
        inputs = [question]
        for tab_en_name in top_k_table_names_en:
            inputs += ["<table>", tab_en_name, "<column>"]
            if isinstance(top_k_col, int):
                inputs += table_names_en2col[tab_en_name][:top_k_col]
            elif top_k_col == 'all':
                inputs += table_names_en2col[tab_en_name]
            else:
                raise NotImplementedError
        _input = " ".join(inputs)

        all_inputs.append(_input)

        meta = {
            "q_id": item["q_id"],
            "question": item["question"],
            "top_k_table_name": [
                [table_names_cn2en[all_names_cn[idx]], all_names_cn[idx]] for idx, _ in sorted_bm25_scores[:top_k_table_names]
            ],
            "top_k_table_col": [
                {tab_en_name: table_names_en2col[tab_en_name] for tab_en_name in top_k_table_names_en}
            ]
        }

        if "db_name" in item:
            all_cls_labels.append(db_vocab[item["db_name"]])
            meta["db_name"] = item["db_name"]

        if "sql_query" in item:
            all_seq_queries.append(item["sql_query"])
            meta["sql_query"] = item["sql_query"]

        if "from" in item:
            meta["sqlite"] = {
                "from": item["from"],
                "select": item["select"],
                "where": item["where"],
                "groupBy": item["groupBy"],
                "having": item["having"],
                "orderBy": item["orderBy"],
                "limit": item["limit"]
            }

        meta_data.append(meta)

    logger.info("Read {} data points".format(len(all_inputs)))

    model_inputs = tokenizer(all_inputs, padding=PaddingStrategy.LONGEST, truncation=True, return_tensors="pt", max_length=max_input_length)
    logger.info(f"Input length: {model_inputs['input_ids'].size(1)}")
    if len(all_seq_queries):
        with tokenizer.as_target_tokenizer():
            model_seq_queries = tokenizer(all_seq_queries, padding=PaddingStrategy.LONGEST, truncation=True,
                                          max_length=max_output_length, return_tensors="pt")
            model_inputs["labels"] = model_seq_queries["input_ids"]
            logger.info(f"Seq2Seq output length: {model_inputs['labels'].size(1)}")
    if len(all_cls_labels):
        model_inputs["cls_labels"] = torch.tensor(all_cls_labels, dtype=torch.long)

    logger.info(f"Saving to {cached_file_path}.")
    torch.save((model_inputs, meta_data), cached_file_path)

    return DictTensorDataset(model_inputs, meta_data)


def read_data_with_table_names_dpr_column_names_sim(file_path, tokenizer: PreTrainedTokenizer, db_info_path: str, dpr_results_path: str,
                                                    top_k_table_names: int = 5, top_k_col: Union[str, int] = 5, resize_vocab: bool = False,
                                                    filter_error: bool = False, add_tab_cn: bool = False,
                                                    max_input_length: int = 128, max_output_length: int = 512):
    if resize_vocab:
        tokenizer.add_tokens(["<table>", "<column>"])
        if add_tab_cn:
            tokenizer.add_tokens(["<cn>"])

    db_info = json.load(open(db_info_path, 'r'))

    table_names_en2col = {}
    all_table_names_en = []
    all_table_names_cn = []
    for db in db_info:
        db_table_names = db["table_name"]
        all_table_names_en.extend([tab_name[0] for tab_name in db_table_names])
        all_table_names_cn.extend([tab_name[1] for tab_name in db_table_names])
        for tab in db["column_info"]:
            tab_en_name = tab["table"]
            tab_col_en = tab["columns"][2:]  # 忽略 `*` 和 `id` 两列
            table_names_en2col[tab_en_name] = tab_col_en

    dpr_results = json.load(open(dpr_results_path, 'r'))

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))
    if filter_error:
        tmp = sql_query_filter(data)
        logger.info("Amount of filtered data: {} / {}".format(len(tmp), len(data)))
        data = tmp

    all_inputs = []
    all_seq_queries = []
    all_cls_labels = []
    meta_data = []

    for item_id, item in enumerate(tqdm(data, desc="Processing data", total=len(data))):
        question = item["question"]

        top_k_table_names_en = [all_table_names_en[idx] for idx in dpr_results[item_id][:top_k_table_names]]
        top_k_table_names_cn = [all_table_names_cn[idx] for idx in dpr_results[item_id][:top_k_table_names]]

        # _input = " ".join([question, "<tables>"] + top_k_table_names_en)
        inputs = [question]
        for tab_en_name, tab_cn_name in zip(top_k_table_names_en, top_k_table_names_cn):
            if add_tab_cn:
                inputs += ["<table>", tab_en_name, "<cn>", tab_cn_name, "<column>"]
            else:
                inputs += ["<table>", tab_en_name, "<column>"]
            if isinstance(top_k_col, int):
                inputs += table_names_en2col[tab_en_name][:top_k_col]
            elif top_k_col == 'all':
                inputs += table_names_en2col[tab_en_name]
            else:
                raise NotImplementedError
        _input = " ".join(inputs)
        all_inputs.append(_input)

        meta = {
            "q_id": item["q_id"],
            "question": item["question"],
            "top_k_table_name": [
                [name_cn, name_en] for name_cn, name_en in zip(top_k_table_names_cn, top_k_table_names_en)
            ],
            "top_k_table_col": [
                {tab_en_name: table_names_en2col[tab_en_name] for tab_en_name in top_k_table_names_en}
            ]
        }

        if "db_name" in item:
            all_cls_labels.append(db_vocab[item["db_name"]])
            meta["db_name"] = item["db_name"]

        if "sql_query" in item:
            all_seq_queries.append(item["sql_query"])
            meta["sql_query"] = item["sql_query"]

        if "from" in item:
            meta["sqlite"] = {
                "from": item["from"],
                "select": item["select"],
                "where": item["where"],
                "groupBy": item["groupBy"],
                "having": item["having"],
                "orderBy": item["orderBy"],
                "limit": item["limit"]
            }

        meta_data.append(meta)

    logger.info("Read {} data points".format(len(all_inputs)))

    model_inputs = tokenizer(all_inputs, padding=PaddingStrategy.LONGEST, truncation=True, return_tensors="pt", max_length=max_input_length)
    logger.info(f"Input length: {model_inputs['input_ids'].size(1)}")
    if len(all_seq_queries):
        with tokenizer.as_target_tokenizer():
            model_seq_queries = tokenizer(all_seq_queries, padding=PaddingStrategy.LONGEST, truncation=True,
                                          max_length=max_output_length, return_tensors="pt")
            model_inputs["labels"] = model_seq_queries["input_ids"]
            logger.info(f"Seq2Seq output length: {model_inputs['labels'].size(1)}")
    if len(all_cls_labels):
        model_inputs["cls_labels"] = torch.tensor(all_cls_labels, dtype=torch.long)

    return DictTensorDataset(model_inputs, meta_data)


def read_data_with_table_names_dpr_column_names_sim_w_aug(file_path, tokenizer: PreTrainedTokenizer, db_info_path: str, kb_file: str,
                                                          dpr_results_path: str, top_k_table_names: int = 5,
                                                          top_k_col: Union[str, int] = 5, resize_vocab: bool = False,
                                                          filter_error: bool = False, add_tab_cn: bool = False, aug_tokenize: bool = False,
                                                          max_input_length: int = 128, max_output_length: int = 512):
    if resize_vocab:
        tokenizer.add_tokens(["<table>", "<column>"])
        if add_tab_cn:
            tokenizer.add_tokens(["<cn>"])

    db_info = json.load(open(db_info_path, 'r'))

    table_names_en2col = {}
    all_table_names_en = []
    all_table_names_cn = []
    for db in db_info:
        db_table_names = db["table_name"]
        all_table_names_en.extend([tab_name[0] for tab_name in db_table_names])
        all_table_names_cn.extend([tab_name[1] for tab_name in db_table_names])
        for tab in db["column_info"]:
            tab_en_name = tab["table"]
            tab_col_en = tab["columns"][2:]  # 忽略 `*` 和 `id` 两列
            table_names_en2col[tab_en_name] = tab_col_en

    dpr_results = json.load(open(dpr_results_path, 'r'))

    logger.info("Reading knowledge base from {}".format(kb_file))
    kb = get_ent_attr_alias(kb_file)

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))
    if filter_error:
        tmp = sql_query_filter(data)
        logger.info("Amount of filtered data: {} / {}".format(len(tmp), len(data)))
        data = tmp

    data_augmentations = augmentation_with_ent_attr_alias(data, kb, tokenize=aug_tokenize)
    for aug in data_augmentations:
        orig_item_id = aug["orig_index"]
        dpr_results.append(dpr_results[orig_item_id])
    data.extend(data_augmentations)
    assert len(dpr_results) == len(data)

    all_inputs = []
    all_seq_queries = []
    all_cls_labels = []
    meta_data = []

    for item_id, item in enumerate(tqdm(data, desc="Processing data", total=len(data))):
        question = item["question"]

        top_k_table_names_en = [all_table_names_en[idx] for idx in dpr_results[item_id][:top_k_table_names]]
        top_k_table_names_cn = [all_table_names_cn[idx] for idx in dpr_results[item_id][:top_k_table_names]]

        # _input = " ".join([question, "<tables>"] + top_k_table_names_en)
        inputs = [question]
        for tab_en_name, tab_cn_name in zip(top_k_table_names_en, top_k_table_names_cn):
            if add_tab_cn:
                inputs += ["<table>", tab_en_name, "<cn>", tab_cn_name, "<column>"]
            else:
                inputs += ["<table>", tab_en_name, "<column>"]
            if isinstance(top_k_col, int):
                inputs += table_names_en2col[tab_en_name][:top_k_col]
            elif top_k_col == 'all':
                inputs += table_names_en2col[tab_en_name]
            else:
                raise NotImplementedError
        _input = " ".join(inputs)
        all_inputs.append(_input)

        meta = {
            "q_id": item["q_id"],
            "question": item["question"],
            "top_k_table_name": [
                [name_cn, name_en] for name_cn, name_en in zip(top_k_table_names_cn, top_k_table_names_en)
            ],
            "top_k_table_col": [
                {tab_en_name: table_names_en2col[tab_en_name] for tab_en_name in top_k_table_names_en}
            ]
        }

        if "db_name" in item:
            all_cls_labels.append(db_vocab[item["db_name"]])
            meta["db_name"] = item["db_name"]

        if "sql_query" in item:
            all_seq_queries.append(item["sql_query"])
            meta["sql_query"] = item["sql_query"]

        if "from" in item:
            meta["sqlite"] = {
                "from": item["from"],
                "select": item["select"],
                "where": item["where"],
                "groupBy": item["groupBy"],
                "having": item["having"],
                "orderBy": item["orderBy"],
                "limit": item["limit"]
            }

        meta_data.append(meta)

    logger.info("Read {} data points".format(len(all_inputs)))

    model_inputs = tokenizer(all_inputs, padding=PaddingStrategy.LONGEST, truncation=True, return_tensors="pt", max_length=max_input_length)
    logger.info(f"Input length: {model_inputs['input_ids'].size(1)}")
    if len(all_seq_queries):
        with tokenizer.as_target_tokenizer():
            model_seq_queries = tokenizer(all_seq_queries, padding=PaddingStrategy.LONGEST, truncation=True,
                                          max_length=max_output_length, return_tensors="pt")
            model_inputs["labels"] = model_seq_queries["input_ids"]
            logger.info(f"Seq2Seq output length: {model_inputs['labels'].size(1)}")
    if len(all_cls_labels):
        model_inputs["cls_labels"] = torch.tensor(all_cls_labels, dtype=torch.long)

    return DictTensorDataset(model_inputs, meta_data)


def get_ent_attr_alias(kb_file: str):
    data = json.load(open(kb_file, 'r'))

    kb = {}
    for ent in data:
        ent_name = ent["entityName"]
        ent_type = ent["entityType"]
        for attr in ent["entityAttributes"]:
            attr_name = attr["attrName"]
            attr_type = attr["attrType"]
            attr_alias = attr["attrAlias"]
            if attr_name not in kb:
                kb[attr_name] = {
                    "alias": attr_alias,
                    "type": attr_type,
                    "entity_name": ent_name,
                    "entity_type": ent_type
                }
            else:
                print(f"Repeat attribute: {attr_name}")
                print(f"{kb[attr_name]}")
                print(f"{attr_alias}, {attr_type}, {ent_name}, {ent_type}")

    return kb


def augmentation_with_ent_attr_alias(data: List[Dict[str, Any]], kb: Dict[str, Any], tokenize: bool = False):
    data_augmentations = []
    for item_id, item in enumerate(data):
        q_id = item["q_id"]
        question = item["question"]
        if tokenize:
            question = list(jieba.cut(question))
        cnt = 0
        for attr_name in kb:
            alias = [attr_name] + kb[attr_name]["alias"]
            for i, ali in enumerate(alias):
                if ali in question:
                    for j, ali_rep in enumerate(alias):
                        if i == j:
                            continue
                        tmp = copy.deepcopy(item)
                        if not tokenize:
                            tmp["question"] = question.replace(ali, ali_rep)
                        else:
                            tmp_q = " ".join([ali_rep if x == ali else x for x in question])
                            tmp["question"] = tmp_q
                        tmp["q_id"] = f"{q_id}_{cnt}"
                        tmp["ref_attr"] = ali
                        tmp["orig_index"] = item_id
                        data_augmentations.append(tmp)
                    break

    logger.info(f"Augmented data samples: {len(data_augmentations)}")
    return data_augmentations


def get_data_augmentations(data, kb_path: str, tokenize: bool = False):
    kb = get_ent_attr_alias(kb_path)
    data_augmentations = augmentation_with_ent_attr_alias(data, kb, tokenize=tokenize)
    return data_augmentations, kb
