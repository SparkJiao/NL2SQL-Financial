import json
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Union, Any, Tuple

import jieba
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy

from data.bm25 import BM25Model
from data.collators import DictTensorDataset
from data.data_utils import tokenizer_get_name
from data.multi_task_grounding import sequence_annotate, check_consistency, remove_out_of_bounds, table_ground_sup, value_ground_sup, \
    column_ground_sup, value_column_match_sup
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

_tokenizer: PreTrainedTokenizer
_db_vocab: Dict[str, int]
_db2tab: Dict[str, List[str]]
_db_col_id2tab_id_col: Dict[str, List[List[Union[int, str]]]]


def read_db_schema(db_path: str):
    data = json.load(open(db_path, 'r', encoding='utf-8'))

    dbs = []
    db_vocab = {}
    db_col_id2tab_id_col = {}
    db2tab = {}
    for db in data:
        db_name: str = db["db_id"]
        db_vocab[db_name] = len(dbs)
        dbs.append(db_name)

        db_tables = db["table_names"]
        db2tab[db_name] = db_tables

        db_columns = db["column_names"]
        # if db_columns[0][0] == -1:  # Remove [-1, '*']
        #     db_columns = db_columns[1:]
        db_col_id2tab_id_col[db_name] = db_columns
        # for col in db_columns:
        #     tab_id, col_name = col
        #     tab2col[db_tables[tab_id]].append(col_name)

        # We overlook ``column_names_original`` and ``table_names_original`` since we don't know the definitions.

    return db_vocab, db2tab, db_col_id2tab_id_col


def read_tab2col(db_columns: List[List], tabs: List[str]):
    tab2col = defaultdict(list)
    for col in db_columns:
        tab_id, col_name = col
        if tab_id == -1:
            continue
        tab2col[tabs[tab_id]].append(col_name)
    return tab2col


def union_grounding(mentioned_tables: set, mentioned_columns: Dict[str, set], mentioned_values: set,
                    tables: set, columns: Dict[str, set], values: set):
    mentioned_tables.update(tables)
    for tab, tab_columns in columns.items():
        if tab in mentioned_columns:
            mentioned_columns[tab].update(tab_columns)
        else:
            mentioned_columns[tab] = tab_columns
    mentioned_values.update(values)
    return mentioned_tables, mentioned_columns, mentioned_values


def parse_col_unit(col_unit: List, tabs: List[str], col_id2tab_id_col: List[List[Union[int, str]]]):
    if len(col_unit) == 3:
        agg_op_id, col_id, col_is_distinct = col_unit
    elif len(col_unit) == 2:
        agg_op_id, col_id = col_unit
    else:
        raise ValueError(col_unit)
    # assert isinstance(col_id, int), (col_unit, col_id)
    if isinstance(col_id, str):  # A specific case
        return None, None
    col_tab_name = tabs[col_id2tab_id_col[col_id][0]]
    col_name = col_id2tab_id_col[col_id][1]
    return col_tab_name, col_name


def parse_val(val: Union[Dict[str, Any], float, int, str], cond_tab: str, cond_col: str,
              tabs: List[str], col_id2tab_id_col: List[List[Union[int, str]]]):
    mentioned_tables = set()
    mentioned_columns = defaultdict(set)
    mentioned_values = set()

    if isinstance(val, Dict):
        return parse_sql(val, tabs, col_id2tab_id_col)
    elif isinstance(val, int):
        # assert val < len(col_id2tab_id_col), (val, len(col_id2tab_id_col))
        # assert col_id2tab_id_col[val][0] < len(tabs), (col_id2tab_id_col[val][0], len(tabs))

        if val >= len(col_id2tab_id_col):  # Hack here. Treat the out-of-bound value as a true value instead of the index of some column.
            mentioned_values.add((val, cond_tab, cond_col))
        else:
            col_tab_name = tabs[col_id2tab_id_col[val][0]]
            col_name = col_id2tab_id_col[val][1]

            mentioned_tables.add(col_tab_name)
            mentioned_columns[col_tab_name].add(col_name)
    elif isinstance(val, float) or isinstance(val, str):
        # FIXME: Consider normalize 1989.0 to 1989, where only the latter one exists in the query.
        if str(val)[-2:] == '.0':
            # print(val, str(val)[:-2])
            val = str(val)[:-2]
        mentioned_values.add((val, cond_tab, cond_col))
    else:
        raise ValueError(val)

    return mentioned_tables, mentioned_columns, mentioned_values


def parse_condition_group(conditions: List[Union[str, List]], tabs: List[str], col_id2tab_id_col: List[List[Union[int, str]]]):
    mentioned_tables = set()
    mentioned_columns = defaultdict(set)
    mentioned_values = set()

    for cond_id, cond in enumerate(conditions):
        if isinstance(cond, str):
            assert cond in ["and", "or"] and cond_id % 2 == 1, cond
        elif isinstance(cond, List):
            last_tab, last_col = None, None

            not_op, where_ops_id, val_unit, val1, val2 = cond
            unit_op_id, col_unit1, col_unit2 = val_unit
            assert len(cond) == 5, cond
            assert len(val_unit) == 3, val_unit
            # assert col_unit1 is None or len(col_unit1) == 3, col_unit1
            # assert col_unit2 is None or len(col_unit2) == 3, col_unit2
            for col_unit in [col_unit1, col_unit2]:
                if col_unit is not None:
                    col_tab_name, col_name = parse_col_unit(col_unit, tabs, col_id2tab_id_col)
                    if col_tab_name is None or col_name is None:
                        continue

                    mentioned_tables.add(col_tab_name)
                    mentioned_columns[col_tab_name].add(col_name)
                    last_tab = col_tab_name
                    last_col = col_name

            for val in [val1, val2]:
                if val is not None:
                    # FIXME: ``cond_tab`` and ``cond_col`` may both be ``None``
                    mentioned_tables, mentioned_columns, mentioned_values = union_grounding(
                        mentioned_tables, mentioned_columns, mentioned_values, *parse_val(val, last_tab, last_col, tabs, col_id2tab_id_col)
                    )
        else:
            raise ValueError(cond_id, cond)

    return mentioned_tables, mentioned_columns, mentioned_values


def parse_sql(sql: Dict[str, Any], tabs: List[str], col_id2tab_id_col: List[List[Union[int, str]]]):
    """
    In convenient for recursion.
    """
    mentioned_tables = set()
    mentioned_values = set()
    mentioned_columns = defaultdict(set)

    # condition
    conditions = []

    # from
    from_struct = sql["from"]
    table_units = from_struct["table_units"]
    for table_unit in table_units:
        table_type, tab_id = table_unit
        if isinstance(tab_id, dict):
            union_grounding(mentioned_tables, mentioned_columns, mentioned_values,
                            *parse_sql(tab_id, tabs, col_id2tab_id_col))
        elif isinstance(tab_id, int):
            mentioned_tables.add(tabs[tab_id])
        else:
            raise ValueError(tab_id)
    table_conditions = from_struct["conds"]  # ~~Just overlook conditions since we don't need it to parse mentioned tables or columns.~~
    union_grounding(mentioned_tables, mentioned_columns, mentioned_values,
                    *parse_condition_group(table_conditions, tabs, col_id2tab_id_col))

    # select
    select_struct = sql["select"]
    for item in select_struct:
        agg_op_id, val_unit = item
        unit_op_id, col_unit1, col_unit2 = val_unit
        for col_unit in [col_unit1, col_unit2]:
            if col_unit is not None:
                col_tab_name, col_name = parse_col_unit(col_unit, tabs, col_id2tab_id_col)
                if col_tab_name is None or col_name is None:
                    continue

                mentioned_tables.add(col_tab_name)
                mentioned_columns[col_tab_name].add(col_name)

    # where
    where_conditions = sql["where"]
    union_grounding(mentioned_tables, mentioned_columns, mentioned_values,
                    *parse_condition_group(where_conditions, tabs, col_id2tab_id_col))

    return mentioned_tables, mentioned_columns, mentioned_values


def parse_grounding_from_structure(item, db2tab: Dict[str, List[str]], db_col_id2tab_id_col: Dict[str, List[List[Union[int, str]]]]):
    """
    Similar to ``parse_grounding_from_structure`` method in data_utils.py but take single data item as input for reuse.

    ~~The structure information is referenced from https://github.com/taoyds/spider/blob/master/preprocess/parsed_sql_examples.sql~~
    The structure information is referenced from
        https://github.com/PaddlePaddle/Research/blob/master/NLP/Text2SQL-BASELINE/tools/evaluation/text2sql_evaluation.py

    It should be noted that DuSQL doesn't involve db classification and the tables are specific to different dbs. And we didn't check
    if there are the same names of tables across different dbs.
    """
    structure = item["sql"]
    db_name = item["db_id"]

    tabs = db2tab[db_name]
    col_id2tab_id_col = db_col_id2tab_id_col[db_name]

    mentioned_tables, mentioned_columns, mentioned_values = parse_sql(structure, tabs, col_id2tab_id_col)
    return mentioned_tables, mentioned_columns, mentioned_values


def process_init(tokenizer: PreTrainedTokenizer, db_vocab: Dict[str, int], db2tab: Dict[str, List[str]],
                 db_col_id2tab_id_col: Dict[str, List[List[Union[int, str]]]]):
    global _tokenizer
    _tokenizer = tokenizer

    global _db_vocab
    _db_vocab = db_vocab

    global _db2tab
    _db2tab = db2tab

    global _db_col_id2tab_id_col
    _db_col_id2tab_id_col = db_col_id2tab_id_col


def process_single_item(item, top_k_tab: int, top_k_col: Union[str, int]):
    question = item["question"]
    db_name = item["db_id"]

    meta_data = {
        "q_id": item["question_id"],
        "question": question,
        "db_name": db_name
    }

    if "sql" in item and item["sql"]:
        meta_data["sqlite"] = item["sql"]
        mentioned_tables, mentioned_columns, mentioned_values = parse_grounding_from_structure(item,
                                                                                               _db2tab,
                                                                                               _db_col_id2tab_id_col)
        mentioned_tables = list(mentioned_tables)

        columns_dict_list = {}
        for tab, col_set in mentioned_columns.items():
            columns_dict_list[tab] = list(col_set)
        mentioned_columns = columns_dict_list

        mentioned_values = list(mentioned_values)
    else:
        mentioned_tables, mentioned_columns, mentioned_values = [], {}, []

    meta_data["parsing"] = {
        "tables": mentioned_tables,
        "columns": mentioned_columns,
        "values": mentioned_values
    }

    if isinstance(top_k_col, str) and top_k_col == "all":
        top_k_col = 10000

    tables = _db2tab[db_name][:top_k_tab]
    tab2col = read_tab2col(_db_col_id2tab_id_col[db_name], _db2tab[db_name])
    columns = []
    for table in tables:
        columns.append(tab2col[table][:top_k_col])

    question, tab2pos, tab_col2pos, val2pos = sequence_annotate(question, tables, columns, mentioned_values, _tokenizer)

    re_tables, re_columns, re_values = check_consistency(question, tab2pos, tab_col2pos, val2pos, _tokenizer)
    meta_data["debug"] = {
        "question": question,
        "tables": re_tables,
        "columns": re_columns,
        "values": re_values
    }

    return question, tab2pos, tab_col2pos, val2pos, meta_data


def process_single_item_multiple_db(item: Tuple[Dict, List[str]], top_k_tab: int, top_k_col: Union[str, int]):
    item, db_list = item

    question = item["question"]
    # db_name = item["db_id"]

    meta_data = {
        "q_id": item["question_id"].replace("qid", ""),
        "question": question,
        # "db_name": ""
    }

    # if "sql" in item and item["sql"]:
    #     meta_data["sqlite"] = item["sql"]
    #     mentioned_tables, mentioned_columns, mentioned_values = parse_grounding_from_structure(item,
    #                                                                                            _db2tab,
    #                                                                                            _db_col_id2tab_id_col)
    #     mentioned_tables = list(mentioned_tables)
    #
    #     columns_dict_list = {}
    #     for tab, col_set in mentioned_columns.items():
    #         columns_dict_list[tab] = list(col_set)
    #     mentioned_columns = columns_dict_list
    #
    #     mentioned_values = list(mentioned_values)
    # else:
    mentioned_tables, mentioned_columns, mentioned_values = [], {}, []

    meta_data["parsing"] = {
        "tables": mentioned_tables,
        "columns": mentioned_columns,
        "values": mentioned_values
    }

    if isinstance(top_k_col, str) and top_k_col == "all":
        top_k_col = 10000

    tables = []
    columns = []
    for db_name in db_list:
        db_tables = _db2tab[db_name][:top_k_tab]
        tables.extend(db_tables)

        db_tab2col = read_tab2col(_db_col_id2tab_id_col[db_name], _db2tab[db_name])
        for table in db_tables:
            columns.append(db_tab2col[table][:top_k_col])

        assert len(tables) == len(columns)

    question, tab2pos, tab_col2pos, val2pos = sequence_annotate(question, tables, columns, mentioned_values, _tokenizer)

    re_tables, re_columns, re_values = check_consistency(question, tab2pos, tab_col2pos, val2pos, _tokenizer)
    meta_data["debug"] = {
        "question": question,
        "tables": re_tables,
        "columns": re_columns,
        "values": re_values
    }

    return question, tab2pos, tab_col2pos, val2pos, meta_data


def read_data_w_ground(file_path: str, tokenizer: PreTrainedTokenizer, db_info_path: str,
                       top_k_tab: int = 5, top_k_col: Union[str, int] = 5,
                       max_input_length: int = 512, max_output_length: int = 128,
                       num_workers: int = 4):
    tokenizer_name = tokenizer_get_name(tokenizer)
    file_suffix = f"{tokenizer_name}_{db_info_path.split('/')[-1].replace('.json', '')}_{top_k_tab}_{top_k_col}_{max_input_length}_" \
                  f"{max_output_length}_du_sql_gd"

    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        model_inputs, all_meta_data = torch.load(cached_file_path)
        return DictTensorDataset(model_inputs, all_meta_data)

    tokenizer.add_tokens(["<table>", "<column>"])

    db_vocab, db2tab, db_col_id2tab_id_col = read_db_schema(db_info_path)

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    # process_init(tokenizer, db_vocab, db2tab, db_col_id2tab_id_col)
    all_questions, all_tab2pos, all_tab_col2pos, all_val2pos, all_meta_data = [], [], [], [], []
    # for item in tqdm(data, total=len(data)):
    #     res = process_single_item(item, top_k_tab, top_k_col)
    #     all_questions.append(res[0])
    #     all_tab2pos.append(res[1])
    #     all_tab_col2pos.append(res[2])
    #     all_val2pos.append(res[3])
    #     all_meta_data.append(res[4])
    with Pool(num_workers, initializer=process_init, initargs=(tokenizer, db_vocab, db2tab, db_col_id2tab_id_col)) as p:
        _annotate = partial(process_single_item, top_k_tab=top_k_tab, top_k_col=top_k_col)
        results = list(tqdm(
            p.imap(_annotate, data, chunksize=32),
            total=len(data)
        ))
    for res in results:
        all_questions.append(res[0])
        all_tab2pos.append(res[1])
        all_tab_col2pos.append(res[2])
        all_val2pos.append(res[3])
        all_meta_data.append(res[4])

    # Obtain basic labels
    all_seq_queries = []
    for item, item_meta in zip(data, all_meta_data):
        if "query" in item and item["query"]:
            all_seq_queries.append(item["query"])
            item_meta["sql_query"] = item["query"]

    # Re-Tokenization
    model_inputs = tokenizer(all_questions,
                             padding=PaddingStrategy.LONGEST,
                             truncation=True,
                             return_tensors="pt",
                             max_length=max_input_length)
    input_length = model_inputs["input_ids"].size(1)
    logger.info(f"Input length: {input_length}")

    if len(all_seq_queries):
        with tokenizer.as_target_tokenizer():
            model_seq_queries = tokenizer(all_seq_queries,
                                          padding=PaddingStrategy.LONGEST,
                                          truncation=True,
                                          max_length=max_output_length,
                                          return_tensors="pt")
            model_inputs["labels"] = model_seq_queries["input_ids"]
            logger.info(f"Seq2Seq output length: {model_inputs['labels'].size(1)}")

    # Remove out-of-bounds positions caused by truncation.
    all_tab2pos = [remove_out_of_bounds(input_length, item_tab2pos) for item_tab2pos in all_tab2pos]
    for item_tab_col2pos in all_tab_col2pos:
        for tab, col2pos in item_tab_col2pos.items():
            col2pos = remove_out_of_bounds(input_length, col2pos)
            item_tab_col2pos[tab] = col2pos
    all_val2pos = [remove_out_of_bounds(input_length, item_val2pos) for item_val2pos in all_val2pos]

    # Obtain the indices of tables, columns, and values.
    max_table_num = max(map(lambda x: len(x), all_tab2pos))
    tab_pos_index = torch.zeros(len(data), max_table_num, 2, dtype=torch.long)
    tab_mask = torch.zeros(len(data), max_table_num, dtype=torch.long)
    logger.info(f"Max table num: {max_table_num}")
    for item_id, item_tab2pos in enumerate(all_tab2pos):
        tables = [(s, e - 1) for s, e in item_tab2pos.values()]
        tab_pos_index[item_id, :len(tables)] = torch.tensor(tables, dtype=torch.long)
        tab_mask[item_id, :len(item_tab2pos)] = 1

    max_column_num = 0
    for item_tab_col2pos in all_tab_col2pos:
        column_num = sum(map(len, item_tab_col2pos.values()))
        max_column_num = max(max_column_num, column_num)
    col_pos_index = torch.zeros(len(data), max_column_num, 2, dtype=torch.long)
    col_mask = torch.zeros(len(data), max_column_num, dtype=torch.long)
    logger.info(f"Max column num: {max_column_num}")
    for item_id, item_tab_col2pos in enumerate(all_tab_col2pos):
        columns = []
        for tab, col2pos in item_tab_col2pos.items():
            # columns.extend(col2pos.values())
            columns.extend([(s, e - 1) for s, e in col2pos.values()])
        col_pos_index[item_id, :len(columns)] = torch.tensor(columns, dtype=torch.long)
        col_mask[item_id, :len(columns)] = 1

    max_val_num = max(map(len, all_val2pos))
    val_pos_index = torch.zeros(len(data), max_val_num, 2, dtype=torch.long)
    val_mask = torch.zeros(len(data), max_val_num, dtype=torch.long)
    logger.info(f"Max value num: {max_val_num}")
    for item_id, item_val2pos in enumerate(all_val2pos):
        values = [(s, e - 1) for s, e in item_val2pos.values()]
        if len(values) == 0:
            continue
        val_pos_index[item_id, :len(values)] = torch.tensor(values, dtype=torch.long)
        val_mask[item_id, :len(values)] = 1

    model_inputs["tab_index"] = tab_pos_index
    model_inputs["tab_mask"] = tab_mask
    model_inputs["col_index"] = col_pos_index
    model_inputs["col_mask"] = col_mask
    model_inputs["val_index"] = val_pos_index
    model_inputs["val_mask"] = val_mask

    # Get labels for grounding
    all_tab_ground_labels = torch.zeros(len(data), max_table_num, dtype=torch.long)
    for item_id, (item_tab2pos, item_meta) in enumerate(zip(all_tab2pos, all_meta_data)):
        item_tab_labels = table_ground_sup(item_tab2pos, item_meta["parsing"]["tables"])
        all_tab_ground_labels[item_id, :len(item_tab_labels)] = torch.tensor(item_tab_labels, dtype=torch.long)
        assert len(item_tab_labels) == tab_mask[item_id].sum().item()

    all_col_ground_labels = torch.zeros(len(data), max_column_num, dtype=torch.long)
    for item_id, (item_tab_col2pos, item_meta) in enumerate(zip(all_tab_col2pos, all_meta_data)):
        res = column_ground_sup(item_tab_col2pos, item_meta["parsing"]["columns"])
        columns = []
        for column_labels in res:
            columns.extend(column_labels)
        assert len(columns) == col_mask[item_id].sum().item()
        all_col_ground_labels[item_id, :len(columns)] = torch.tensor(columns, dtype=torch.long)

    all_val_ground_labels = [
        value_ground_sup(input_ids, item_val2pos) for input_ids, item_val2pos in zip(model_inputs["input_ids"], all_val2pos)
    ]
    all_val_ground_labels = torch.tensor(all_val_ground_labels, dtype=torch.long)
    assert all_val_ground_labels.size() == model_inputs["input_ids"].size()

    all_val_col_match_labels = torch.zeros(len(data), max_val_num, dtype=torch.long).fill_(-1)
    for item_id, (item_meta, item_val2pos, item_tab_col2pos) in enumerate(zip(all_meta_data, all_val2pos, all_tab_col2pos)):
        item_val_col_matching = value_column_match_sup(item_meta["parsing"]["values"], item_val2pos, item_tab_col2pos)
        assert len(item_val_col_matching) == len(all_val2pos[item_id])
        all_val_col_match_labels[item_id, :len(item_val_col_matching)] = torch.tensor(item_val_col_matching, dtype=torch.long)

    all_tab_ground_labels[~(tab_mask.bool())] = -1
    all_col_ground_labels[~(col_mask.bool())] = -1
    all_val_ground_labels[~(model_inputs["attention_mask"].bool())] = -1

    model_inputs["tab_labels"] = all_tab_ground_labels
    model_inputs["col_labels"] = all_col_ground_labels
    model_inputs["val_labels"] = all_val_ground_labels
    model_inputs["val_col_match_labels"] = all_val_col_match_labels

    # For debug
    num_values = val_mask.sum()
    num_val_col_mat_pairs = (all_val_col_match_labels != -1).sum()
    logger.info(f"Number of total parsed values in dataset: {num_values}.")
    logger.info(f"Number of total parsed matched value-column pairs in dataset: {num_val_col_mat_pairs}.")

    torch.save((model_inputs, all_meta_data), cached_file_path)

    return DictTensorDataset(model_inputs, all_meta_data)


def retrieve_db_bm25(db_info: List, data: List[Dict], top_k: int = 3):
    # Read database information to build index
    db_document = []
    db_ids = []
    for item in tqdm(db_info, desc="Building BM25 model", total=len(db_info)):
        # db_item = item["db_id"]
        db_item = list(jieba.cut(item["db_id"]))
        tables = item["table_names"]
        table2col = defaultdict(list)
        for col_id, col_name in item["column_names"]:
            if col_id == -1:
                continue
            table2col[tables[col_id]].append(col_name)
        for tab_name, col_name_ls in table2col.items():
            # db_item = " ".join([db_item, tab_name] + col_name_ls)
            db_item.extend(list(jieba.cut(tab_name)))
            for col in col_name_ls:
                db_item.extend(list(jieba.cut(col)))
        db_document.append(db_item)
        db_ids.append(item["db_id"])

    bm25_model = BM25Model(db_document)

    # Retrieve top-k databases according to question.
    q_top_k_db = []
    for item in tqdm(data, desc="Processing data", total=len(data)):
        # db_id = item["db_id"]
        question = item["question"]

        ques_words = list(jieba.cut(question))
        bm25_scores = [(idx, _score) for idx, _score in enumerate(bm25_model.get_documents_score(ques_words))]
        sorted_bm25_scores = sorted(bm25_scores, key=lambda x: x[1], reverse=True)

        top_k_db = [db_ids[idx] for idx, _ in sorted_bm25_scores[:top_k]]
        q_top_k_db.append(top_k_db)

    return q_top_k_db


def read_data_w_ground_db_bm25(file_path: str, tokenizer: PreTrainedTokenizer, db_info_path: str,
                               top_k_db: int = 3, top_k_tab: int = 5, top_k_col: Union[str, int] = 5,
                               max_input_length: int = 512, max_output_length: int = 128,
                               num_workers: int = 4):
    """
    For inference only.
    """
    tokenizer_name = tokenizer_get_name(tokenizer)
    file_suffix = f"{tokenizer_name}_{db_info_path.split('/')[-1].replace('.json', '')}_{top_k_tab}_{top_k_col}_{max_input_length}_" \
                  f"{max_output_length}_du_sql_gd_db_bm25_infer"

    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        model_inputs, all_meta_data = torch.load(cached_file_path)
        return DictTensorDataset(model_inputs, all_meta_data)

    tokenizer.add_tokens(["<table>", "<column>"])

    db_info = json.load(open(db_info_path, 'r'))
    db_vocab, db2tab, db_col_id2tab_id_col = read_db_schema(db_info_path)

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    q_top_k_db = retrieve_db_bm25(db_info, data, top_k=top_k_db)

    # process_init(tokenizer, db_vocab, db2tab, db_col_id2tab_id_col)
    all_questions, all_tab2pos, all_tab_col2pos, all_val2pos, all_meta_data = [], [], [], [], []
    with Pool(num_workers, initializer=process_init, initargs=(tokenizer, db_vocab, db2tab, db_col_id2tab_id_col)) as p:
        _annotate = partial(process_single_item_multiple_db, top_k_tab=top_k_tab, top_k_col=top_k_col)
        results = list(tqdm(
            p.imap(_annotate, zip(data, q_top_k_db), chunksize=32),
            total=len(data)
        ))
    for res in results:
        all_questions.append(res[0])
        all_tab2pos.append(res[1])
        all_tab_col2pos.append(res[2])
        all_val2pos.append(res[3])
        all_meta_data.append(res[4])

    # Obtain basic labels
    all_seq_queries = []
    # for item, item_meta in zip(data, all_meta_data):
    #     if "query" in item and item["query"]:
    #         all_seq_queries.append(item["query"])
    #         item_meta["sql_query"] = item["query"]

    # Re-Tokenization
    model_inputs = tokenizer(all_questions,
                             padding=PaddingStrategy.LONGEST,
                             truncation=True,
                             return_tensors="pt",
                             max_length=max_input_length)
    input_length = model_inputs["input_ids"].size(1)
    logger.info(f"Input length: {input_length}")

    if len(all_seq_queries):
        with tokenizer.as_target_tokenizer():
            model_seq_queries = tokenizer(all_seq_queries,
                                          padding=PaddingStrategy.LONGEST,
                                          truncation=True,
                                          max_length=max_output_length,
                                          return_tensors="pt")
            model_inputs["labels"] = model_seq_queries["input_ids"]
            logger.info(f"Seq2Seq output length: {model_inputs['labels'].size(1)}")

    # Remove out-of-bounds positions caused by truncation.
    all_tab2pos = [remove_out_of_bounds(input_length, item_tab2pos) for item_tab2pos in all_tab2pos]
    for item_tab_col2pos in all_tab_col2pos:
        for tab, col2pos in item_tab_col2pos.items():
            col2pos = remove_out_of_bounds(input_length, col2pos)
            item_tab_col2pos[tab] = col2pos
    all_val2pos = [remove_out_of_bounds(input_length, item_val2pos) for item_val2pos in all_val2pos]

    # Obtain the indices of tables, columns, and values.
    max_table_num = max(map(lambda x: len(x), all_tab2pos))
    tab_pos_index = torch.zeros(len(data), max_table_num, 2, dtype=torch.long)
    tab_mask = torch.zeros(len(data), max_table_num, dtype=torch.long)
    logger.info(f"Max table num: {max_table_num}")
    for item_id, item_tab2pos in enumerate(all_tab2pos):
        tables = [(s, e - 1) for s, e in item_tab2pos.values()]
        tab_pos_index[item_id, :len(tables)] = torch.tensor(tables, dtype=torch.long)
        tab_mask[item_id, :len(item_tab2pos)] = 1

    max_column_num = 0
    for item_tab_col2pos in all_tab_col2pos:
        column_num = sum(map(len, item_tab_col2pos.values()))
        max_column_num = max(max_column_num, column_num)
    col_pos_index = torch.zeros(len(data), max_column_num, 2, dtype=torch.long)
    col_mask = torch.zeros(len(data), max_column_num, dtype=torch.long)
    logger.info(f"Max column num: {max_column_num}")
    for item_id, item_tab_col2pos in enumerate(all_tab_col2pos):
        columns = []
        for tab, col2pos in item_tab_col2pos.items():
            # columns.extend(col2pos.values())
            columns.extend([(s, e - 1) for s, e in col2pos.values()])
        col_pos_index[item_id, :len(columns)] = torch.tensor(columns, dtype=torch.long)
        col_mask[item_id, :len(columns)] = 1

    max_val_num = max(map(len, all_val2pos))
    val_pos_index = torch.zeros(len(data), max_val_num, 2, dtype=torch.long)
    val_mask = torch.zeros(len(data), max_val_num, dtype=torch.long)
    logger.info(f"Max value num: {max_val_num}")
    for item_id, item_val2pos in enumerate(all_val2pos):
        values = [(s, e - 1) for s, e in item_val2pos.values()]
        if len(values) == 0:
            continue
        val_pos_index[item_id, :len(values)] = torch.tensor(values, dtype=torch.long)
        val_mask[item_id, :len(values)] = 1

    model_inputs["tab_index"] = tab_pos_index
    model_inputs["tab_mask"] = tab_mask
    model_inputs["col_index"] = col_pos_index
    model_inputs["col_mask"] = col_mask
    model_inputs["val_index"] = val_pos_index
    model_inputs["val_mask"] = val_mask

    # Get labels for grounding
    all_tab_ground_labels = torch.zeros(len(data), max_table_num, dtype=torch.long)
    for item_id, (item_tab2pos, item_meta) in enumerate(zip(all_tab2pos, all_meta_data)):
        item_tab_labels = table_ground_sup(item_tab2pos, item_meta["parsing"]["tables"])
        all_tab_ground_labels[item_id, :len(item_tab_labels)] = torch.tensor(item_tab_labels, dtype=torch.long)
        assert len(item_tab_labels) == tab_mask[item_id].sum().item()

    all_col_ground_labels = torch.zeros(len(data), max_column_num, dtype=torch.long)
    for item_id, (item_tab_col2pos, item_meta) in enumerate(zip(all_tab_col2pos, all_meta_data)):
        res = column_ground_sup(item_tab_col2pos, item_meta["parsing"]["columns"])
        columns = []
        for column_labels in res:
            columns.extend(column_labels)
        assert len(columns) == col_mask[item_id].sum().item()
        all_col_ground_labels[item_id, :len(columns)] = torch.tensor(columns, dtype=torch.long)

    all_val_ground_labels = [
        value_ground_sup(input_ids, item_val2pos) for input_ids, item_val2pos in zip(model_inputs["input_ids"], all_val2pos)
    ]
    all_val_ground_labels = torch.tensor(all_val_ground_labels, dtype=torch.long)
    assert all_val_ground_labels.size() == model_inputs["input_ids"].size()

    all_val_col_match_labels = torch.zeros(len(data), max_val_num, dtype=torch.long).fill_(-1)
    for item_id, (item_meta, item_val2pos, item_tab_col2pos) in enumerate(zip(all_meta_data, all_val2pos, all_tab_col2pos)):
        item_val_col_matching = value_column_match_sup(item_meta["parsing"]["values"], item_val2pos, item_tab_col2pos)
        assert len(item_val_col_matching) == len(all_val2pos[item_id])
        all_val_col_match_labels[item_id, :len(item_val_col_matching)] = torch.tensor(item_val_col_matching, dtype=torch.long)

    all_tab_ground_labels[~(tab_mask.bool())] = -1
    all_col_ground_labels[~(col_mask.bool())] = -1
    all_val_ground_labels[~(model_inputs["attention_mask"].bool())] = -1

    model_inputs["tab_labels"] = all_tab_ground_labels
    model_inputs["col_labels"] = all_col_ground_labels
    model_inputs["val_labels"] = all_val_ground_labels
    model_inputs["val_col_match_labels"] = all_val_col_match_labels

    # For debug
    num_values = val_mask.sum()
    num_val_col_mat_pairs = (all_val_col_match_labels != -1).sum()
    logger.info(f"Number of total parsed values in dataset: {num_values}.")
    logger.info(f"Number of total parsed matched value-column pairs in dataset: {num_val_col_mat_pairs}.")

    torch.save((model_inputs, all_meta_data), cached_file_path)

    return DictTensorDataset(model_inputs, all_meta_data)
