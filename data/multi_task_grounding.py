import copy
import json
from typing import Union, List, Dict, Any, Tuple, Optional
from multiprocessing import Pool

import torch
import jieba
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizer, T5Tokenizer, T5TokenizerFast, MT5Tokenizer, MT5TokenizerFast
from transformers.tokenization_utils import PaddingStrategy

from data.bm25 import BM25Model
from data.collators import DictTensorDataset
from data.seq2seq import get_ent_attr_alias, augmentation_with_ent_attr_alias
from data.data_utils import parse_grounding_from_structure
from general_util.logger import get_child_logger
from data.entity_label_utils import load_kb_entity_enum, annotate_entity, entity_labeling_init

logger = get_child_logger("Seq2SeqData")

db_vocab = {
    "ccks_stock": 0,
    "ccks_fund": 1,
    "ccks_macro": 2,
}

_tokenizer: PreTrainedTokenizer = None


def get_start_offset(tokenizer: PreTrainedTokenizer):
    if any(
            isinstance(tokenizer, cls) for cls in [
                T5Tokenizer,
                T5TokenizerFast,
                MT5Tokenizer,
                MT5TokenizerFast,
            ]
    ):
        return 0
    else:
        raise NotImplementedError


def annotate_span(tokens: List[str], span: str, tokenizer: PreTrainedTokenizer):
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            tmp = tokenizer.convert_tokens_to_string(tokens[i: j + 1])
            if tmp == span:
                return True, i, j + 1
    return False, -1, -1


def sequence_annotate(question: str, table_names: List[str], column_names: List[List[str]], values: List[Tuple[str, str, str]],
                      tokenizer: PreTrainedTokenizer):
    """
    Annotate the positions of tables and columns in the tokenized single input sequence.

    A input sequence is organized like:
    <s> question <table> table_1 <column> col_1 col_2 ... <table> table_2 <column> col_1 col_2 ... <table> ...

    Make sure that you have added the extra tokens, i.e., <table>, <column>, <cn> into the vocabulary of the tokenizer.

    Args:
        question (`str`):
            The question for semantic parsing.
        table_names (`List[str]`):
            The headers of the input tables. Each string is a header.
        column_names (`List[List[str]]`):
            The list of columns in each table.
        values (`List[Tuple[str, str, str]]`):
            The list of values involved in this question. Each value is in the form of `(value, table, column)`.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer for text tokenization.
            Currently, we only consider the tokenizer of generative models.
    """
    # added_num = tokenizer.add_tokens(["<table>", "<column>"])
    # logger.info(f"{added_num} extra tokens added to {tokenizer.__class__.__name__}.")
    tokens = tokenizer.tokenize(question)

    _offset = get_start_offset(tokenizer)

    value2pos = {}
    for val, _, _ in values:
        flag, s, e = annotate_span(tokens, val, tokenizer)
        if flag:
            value2pos[val] = (s + _offset, e + _offset)

    tab2pos = {}
    tab_col2pos = {}

    for tab, tab_columns in zip(table_names, column_names):
        sub_tokens = tokenizer.tokenize("<table>")
        assert len(sub_tokens) == 1
        tokens.extend(sub_tokens)

        s = len(tokens) + _offset
        tokens.extend(tokenizer.tokenize(tab))
        e = len(tokens) + _offset
        tab2pos[tab] = (s, e)

        sub_tokens = tokenizer.tokenize("<column>")
        assert len(sub_tokens) == 1
        tokens.extend(sub_tokens)

        tab_col2pos[tab] = {}
        for col in tab_columns:
            s = len(tokens) + _offset
            tokens.extend(tokenizer.tokenize(f" {col}"))
            e = len(tokens) + _offset
            tab_col2pos[tab][col] = (s, e)

    return tokenizer.convert_tokens_to_string(tokens), tab2pos, tab_col2pos, value2pos


def sequence_annotate_prefix(question: str, table_names: List[str], column_names: List[List[str]],
                             values: List[Tuple[str, str, str]], tokenizer: PreTrainedTokenizer,
                             prefix: Optional[str] = None, suffix: Optional[str] = None, ):
    """
    Annotate the positions of tables and columns in the tokenized single input sequence.

    A input sequence is organized like:
    <s> question <table> table_1 <column> col_1 col_2 ... <table> table_2 <column> col_1 col_2 ... <table> ...

    Make sure that you have added the extra tokens, i.e., <table>, <column>, <cn> into the vocabulary of the tokenizer.

    Args:
        question (`str`):
            The question for semantic parsing.
        table_names (`List[str]`):
            The headers of the input tables. Each string is a header.
        column_names (`List[List[str]]`):
            The list of columns in each table.
        values (`List[Tuple[str, str, str]]`):
            The list of values involved in this question. Each value is in the form of `(value, table, column)`.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer for text tokenization.
            Currently, we only consider the tokenizer of generative models.
        prefix (`str`):
            The prefix in inputs, which are the concatenated retrieved examples here.
        suffix (`str`):
            The suffix in inputs, which are the concatenated retrieved examples here.
            Either prefix or suffix has to be set.
    """
    # added_num = tokenizer.add_tokens(["<table>", "<column>"])
    # logger.info(f"{added_num} extra tokens added to {tokenizer.__class__.__name__}.")
    tokens = tokenizer.tokenize(question)

    state = int(prefix is not None) + int(suffix is not None)
    assert state == 1, "Either suffix or prefix can be set to None."
    prefix_tokens = suffix_tokens = None
    if prefix is not None:
        prefix_tokens = tokenizer.tokenize(prefix)

        _offset = get_start_offset(tokenizer) + len(prefix_tokens)
    elif suffix is not None:
        suffix_tokens = tokenizer.tokenize(suffix)

        _offset = get_start_offset(tokenizer)
    else:
        raise RuntimeError("Either suffix or prefix can be set to None.")

    value2pos = {}
    for val, _, _ in values:
        flag, s, e = annotate_span(tokens, val, tokenizer)
        if flag:
            value2pos[val] = (s + _offset, e + _offset)

    tab2pos = {}
    tab_col2pos = {}

    for tab, tab_columns in zip(table_names, column_names):
        sub_tokens = tokenizer.tokenize("<table>")
        assert len(sub_tokens) == 1
        tokens.extend(sub_tokens)

        s = len(tokens) + _offset
        tokens.extend(tokenizer.tokenize(tab))
        e = len(tokens) + _offset
        tab2pos[tab] = (s, e)

        sub_tokens = tokenizer.tokenize("<column>")
        assert len(sub_tokens) == 1
        tokens.extend(sub_tokens)

        tab_col2pos[tab] = {}
        for col in tab_columns:
            s = len(tokens) + _offset
            tokens.extend(tokenizer.tokenize(f" {col}"))
            e = len(tokens) + _offset
            tab_col2pos[tab][col] = (s, e)

    if prefix_tokens is not None:
        new_question = tokenizer.convert_tokens_to_string(prefix_tokens + tokens)
    elif suffix_tokens is not None:
        new_question = tokenizer.convert_tokens_to_string(tokens + suffix_tokens)
    else:
        raise RuntimeError("Either suffix or prefix can be set to None.")

    return new_question, tab2pos, tab_col2pos, value2pos


def check_consistency(question, tab2pos, tab_col2pos, value2pos, tokenizer: PreTrainedTokenizer):
    tokens = tokenizer.tokenize(question)

    tables = []
    for tab, (s, e) in tab2pos.items():
        tables.append(tokenizer.convert_tokens_to_string(tokens[s: e]))
        if tables[-1] != tab:
            # logger.warning(f"Inconsistent table name: {question}\t{tab}\t{tables[-1]}")
            pass

    columns = {}
    for tab, col2pos in tab_col2pos.items():
        columns[tab] = list()
        for col, (s, e) in col2pos.items():
            columns[tab].append(tokenizer.convert_tokens_to_string(tokens[s: e]))
            if col != columns[tab][-1]:
                # logger.warning(f"Inconsistent column name: {question}\t{col}\t{columns[tab][-1]}")
                pass

    values = []
    for val, (s, e) in value2pos.items():
        values.append(tokenizer.convert_tokens_to_string(tokens[s: e]))
        if val != values[-1]:
            # logger.info(f"Inconsistent value: {question}\t{val}\t{values[-1]}")
            pass

    return tables, columns, values


def table_ground_sup(tab2pos: Dict[str, Tuple[int, int]], mentioned_tables: List[str]):
    mentioned_tables_set = set(mentioned_tables)
    tab_exist_labels = []
    for tab in tab2pos.keys():
        if tab in mentioned_tables_set:
            tab_exist_labels.append(1)
        else:
            tab_exist_labels.append(0)

    return tab_exist_labels


def column_ground_sup(tab_col2pos: Dict[str, Dict[str, Tuple[int, int]]], mentioned_columns_dict_list: Dict[str, List[str]]):
    tab_col_exist_labels = []
    for tab, col2pos_list in tab_col2pos.items():
        if tab not in mentioned_columns_dict_list:
            tab_col_exist_labels.append([0] * len(col2pos_list))
        else:
            col_exist_labels = []
            mentioned_col_set = set(mentioned_columns_dict_list[tab])
            for col in col2pos_list.keys():
                if col in mentioned_col_set:
                    col_exist_labels.append(1)
                else:
                    col_exist_labels.append(0)
            tab_col_exist_labels.append(col_exist_labels)

    return tab_col_exist_labels


def value_ground_sup(input_seq: Union[List[str], List[int], Tensor], val2pos: Dict[str, Tuple[int, int]]):
    val_ground_labels = [0] * len(input_seq)

    for val, pos in val2pos.items():
        if pos[0] >= len(val_ground_labels) or pos[1] > len(val_ground_labels):
            continue
        val_ground_labels[pos[0]: pos[1]] = [1] * (pos[1] - pos[0])

    return val_ground_labels


def value_column_match_sup(mentioned_values: List[Tuple[str, str, str]], val2pos: Dict[str, Tuple[int, int]],
                           tab_col2pos: Dict[str, Dict[str, Tuple[int, int]]]):
    val2tab_col = {}
    for val, tab, col in mentioned_values:
        if val in val2pos:
            if val in val2tab_col:
                logger.warning(f"Multiple value table-column mapping: {(val, val2tab_col[val])} || {(val, tab, col)}")
                val2tab_col[val] = (None, None)
            else:
                val2tab_col[val] = (tab, col)
    # print(val2tab_col)

    tab_col2index = {}
    cnt = 0
    for tab, col2pos in tab_col2pos.items():
        tab_col2index[tab] = {}
        for col in col2pos.keys():
            tab_col2index[tab][col] = cnt
            cnt += 1
    # print(tab_col2index)

    val_col_matching_labels = []
    for val in val2pos.keys():
        tab, col = val2tab_col[val]
        if tab is None or col is None:
            val_col_matching_labels.append(-1)  # Ignore index == -1
        elif tab not in tab_col2index or col not in tab_col2index[tab]:
            val_col_matching_labels.append(-1)
        else:
            val_col_matching_labels.append(tab_col2index[tab][col])
    # print(val_col_matching_labels)

    return val_col_matching_labels


def process_init(tokenizer: PreTrainedTokenizer):
    global _tokenizer
    _tokenizer = tokenizer


def _process_single_input(item: Dict[str, Any],
                          top_k_tab_en: List[str],
                          top_k_col_en: List[List[str]],
                          top_k_tab_cn: List[str] = None,
                          top_k_col_cn: List[str] = None,
                          prefix: Optional[str] = None,
                          suffix: Optional[str] = None):
    question = item["question"]
    meta_data = {
        "q_id": item["q_id"],
        "question": question,
    }

    if "from" in item:
        sqlite = {
            "from": item["from"],
            "select": item["select"],
            "where": item["where"],
            "groupBy": item["groupBy"],
            "having": item["having"],
            "orderBy": item["orderBy"],
            "limit": item["limit"]
        }
        meta_data["sqlite"] = sqlite
    else:
        sqlite = None

    if top_k_tab_cn is not None:
        top_k_tab_en = [f"{_en}<cn>{_cn}" for _en, _cn in zip(top_k_tab_en, top_k_tab_cn)]

    if top_k_col_cn is not None:
        for idx, col_en_list in enumerate(top_k_col_en):
            top_k_col_en[idx] = [f"{_en} {_cn}" for _en, _cn in zip(top_k_col_en[idx], top_k_col_cn[idx])]

    # meta_data["top_k_table_name"] = top_k_tab_en
    meta_data["top_k_table_col"] = {
        tab: tab_col for tab, tab_col in zip(top_k_tab_en, top_k_col_en)
    }

    if sqlite is not None:
        mentioned_tables, mentioned_columns, mentioned_values = parse_grounding_from_structure(sqlite)
    else:
        mentioned_tables, mentioned_columns, mentioned_values = [], {}, []

    meta_data["parsing"] = {
        "tables": mentioned_tables,
        "columns": mentioned_columns,
        "values": mentioned_values
    }

    if prefix is not None or suffix is not None:
        question, tab2pos, tab_col2pos, val2pos = sequence_annotate_prefix(question, top_k_tab_en, top_k_col_en,
                                                                           mentioned_values, _tokenizer, prefix=prefix, suffix=suffix)
    else:
        question, tab2pos, tab_col2pos, val2pos = sequence_annotate(question, top_k_tab_en, top_k_col_en, mentioned_values, _tokenizer)

    re_tables, re_columns, re_values = check_consistency(question, tab2pos, tab_col2pos, val2pos, _tokenizer)
    meta_data["debug"] = {
        "question": question,
        "tables": re_tables,
        "columns": re_columns,
        "values": re_values,
    }

    return question, tab2pos, tab_col2pos, val2pos, meta_data


def get_tab_col_en2cn(db_info):
    tab_col_en2cn = {}
    for db in db_info:
        tables = db["table_name"]
        for tab_en, tab_cn in tables:
            tab_col_en2cn[tab_en] = {"cn": tab_cn}

        for column_info in db["column_info"]:
            tab_en = column_info["table"]
            col_en_list = column_info["columns"]
            col_cn_list = column_info["column_chiName"]
            tab_col_en2cn[tab_en]["col"] = {col_en.lower(): col_cn for col_en, col_cn in zip(col_en_list, col_cn_list) if
                                            col_cn is not None}
    return tab_col_en2cn


def get_aligned_examples(item: Dict[str, Any], tab_col_en2cn: Dict[str, Any]):
    if "from" in item:
        sqlite = {
            "from": item["from"],
            "select": item["select"],
            "where": item["where"],
            "groupBy": item["groupBy"],
            "having": item["having"],
            "orderBy": item["orderBy"],
            "limit": item["limit"]
        }

        mentioned_tables, mentioned_columns, _ = parse_grounding_from_structure(sqlite)

        orig_question = question = item["question"]
        # TODO: 用fin_kb.json里定义的同义词一起搜索
        for tab, col_list in mentioned_columns.items():
            # tab_cn = tab_en2cn[tab]
            if tab == "$TEMP":
                if len(mentioned_tables) == 1:
                    tab = mentioned_tables[0]
                else:
                    continue
            tab_cn = tab_col_en2cn[tab]["cn"]
            if tab_cn in question:
                question = question.replace(tab_cn, tab)
            for col in col_list:
                # col_cn = col_en2cn[col]
                # if col == "*":
                #     continue
                if col not in tab_col_en2cn[tab]["col"]:
                    continue
                col_cn = tab_col_en2cn[tab]["col"][col]
                if col_cn in question:
                    question = question.replace(col_cn, col)
        # print(question)
        if question == orig_question:
            return None

        new_item = copy.deepcopy(item)
        new_item["question"] = question
        return new_item
    return None


def remove_out_of_bounds(length, pos_dict: Dict[str, Tuple[int, int]]):
    outputs = {}
    for k, (s, e) in pos_dict.items():
        if s >= length or e > length:
            continue
        outputs[k] = (s, e)
    return outputs


def read_data_w_multi_ground(file_path: str, tokenizer: PreTrainedTokenizer, db_info_path: str, dpr_results_path: str,
                             top_k_tab: int = 5, top_k_col: Union[str, int] = 5, add_tab_cn: bool = False, name_lowercase: bool = False,
                             max_input_length: int = 512, max_output_length: int = 128):
    tokenizer.add_tokens(["<table>", "<column>"])
    if add_tab_cn:
        tokenizer.add_tokens(["<cn>"])

    db_info = json.load(open(db_info_path, 'r'))

    table_names_en2col = {}
    all_table_names_en = []
    all_table_names_cn = []
    for db in db_info:
        db_table_names = db["table_name"]
        # all_table_names_en.extend([tab_name[0] for tab_name in db_table_names])
        # all_table_names_cn.extend([tab_name[1] for tab_name in db_table_names])
        # for tab in db["column_info"]:
        #     tab_en_name = tab["table"]
        #     tab_col_en = tab["columns"][2:]  # 忽略 `*` 和 `id` 两列
        all_table_names_en.extend([tab_name[0] for tab_name in db_table_names])
        all_table_names_cn.extend([tab_name[1] for tab_name in db_table_names])
        for tab in db["column_info"]:
            tab_en_name = tab["table"]
            if name_lowercase:
                tab_col_en = [tmp.lower() for tmp in tab["columns"][2:]]  # 忽略 `*` 和 `id` 两列
            else:
                tab_col_en = tab["columns"][2:]  # 忽略 `*` 和 `id` 两列
            table_names_en2col[tab_en_name] = tab_col_en

    dpr_results = json.load(open(dpr_results_path, 'r'))

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    top_k_tab_en = []
    top_k_tab_cn = []
    top_k_col_en = []
    for item_id, item in enumerate(data):
        item_top_k_tab_en = list(map(lambda x: all_table_names_en[x], dpr_results[item_id][:top_k_tab]))
        top_k_tab_en.append(item_top_k_tab_en)

        if add_tab_cn:
            top_k_tab_cn.append(list(map(lambda x: all_table_names_cn[x], dpr_results[item_id][:top_k_tab])))
        else:
            top_k_tab_cn.append(None)

        if top_k_col == "all":
            item_top_k_col = list(map(lambda x: table_names_en2col[x], item_top_k_tab_en))
        else:
            item_top_k_col = list(map(lambda x: table_names_en2col[x][:top_k_col], item_top_k_tab_en))
        top_k_col_en.append(item_top_k_col)

    process_init(tokenizer)
    all_questions, all_tab2pos, all_tab_col2pos, all_val2pos, all_meta_data = [], [], [], [], []
    for item, item_top_k_tab_en, item_top_k_col_en, item_top_k_tab_cn in tqdm(zip(data, top_k_tab_en, top_k_col_en, top_k_tab_cn),
                                                                              total=len(data)):
        res = _process_single_input(item, item_top_k_tab_en, item_top_k_col_en, item_top_k_tab_cn)
        all_questions.append(res[0])
        all_tab2pos.append(res[1])
        all_tab_col2pos.append(res[2])
        all_val2pos.append(res[3])
        all_meta_data.append(res[4])

    # Obtain basic labels.
    all_cls_labels = []
    all_seq_queries = []
    for item, item_meta in zip(data, all_meta_data):
        if "db_name" in item:
            item_meta["db_name"] = item["db_name"]
            all_cls_labels.append(db_vocab[item["db_name"]])

        if "sql_query" in item:
            item_meta["sql_query"] = item["sql_query"]
            all_seq_queries.append(item["sql_query"])

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

    if len(all_cls_labels):
        model_inputs["cls_labels"] = torch.tensor(all_cls_labels, dtype=torch.long)

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
        # tab_pos_index[item_id, :len(item_tab2pos)] = torch.tensor(list(item_tab2pos.values()))
        tables = [(s, e - 1) for s, e in item_tab2pos.values()]  # FIXME: Fixed at 2022/07/02
        tab_pos_index[item_id, :len(tables)] = torch.tensor(tables, dtype=torch.long)
        tab_mask[item_id, :len(item_tab2pos)] = 1

    max_column_num = 0
    for item_tab_col2pos in all_tab_col2pos:
        column_num = sum(map(len, item_tab_col2pos.values()))
        max_column_num = max(max_column_num, column_num)
    col_pos_index = torch.zeros(len(data), max_column_num, 2, dtype=torch.long)
    col_mask = torch.zeros(len(data), max_column_num, dtype=torch.long)
    for item_id, item_tab_col2pos in enumerate(all_tab_col2pos):
        columns = []
        for tab, col2pos in item_tab_col2pos.items():
            # columns.extend(col2pos.values())
            columns.extend([(s, e - 1) for s, e in col2pos.values()])  # FIXME: Fixed at 2022/07/02
        col_pos_index[item_id, :len(columns)] = torch.tensor(columns, dtype=torch.long)
        col_mask[item_id, :len(columns)] = 1

    model_inputs["tab_index"] = tab_pos_index
    model_inputs["tab_mask"] = tab_mask
    model_inputs["col_index"] = col_pos_index
    model_inputs["col_mask"] = col_mask

    # Get labels for grounding
    all_tab_ground_labels = torch.zeros(len(data), max_table_num, dtype=torch.long)
    for item_id, (item_tab2pos, item_meta) in enumerate(zip(all_tab2pos, all_meta_data)):
        item_tab_labels = table_ground_sup(item_tab2pos, item_meta["parsing"]["tables"])
        all_tab_ground_labels[item_id, :len(item_tab_labels)] = torch.tensor(item_tab_labels, dtype=torch.long)
        assert len(item_tab_labels) == tab_mask[item_id].sum().item()

    # all_tab_ground_labels = [
    #     table_ground_sup(item_tab2pos, item_meta["parsing"]["tables"]) for item_tab2pos, item_meta in zip(all_tab2pos, all_meta_data)
    # ]

    all_col_ground_labels = torch.zeros(len(data), max_column_num, dtype=torch.long)
    for item_id, (item_tab_col2pos, item_meta) in enumerate(zip(all_tab_col2pos, all_meta_data)):
        res = column_ground_sup(item_tab_col2pos, item_meta["parsing"]["columns"])
        columns = []
        for column_labels in res:
            columns.extend(column_labels)
        assert len(columns) == col_mask[item_id].sum().item()
        all_col_ground_labels[item_id, :len(columns)] = torch.tensor(columns, dtype=torch.long)

    # all_col_ground_labels = [
    #     column_ground_sup(item_tab_col2pos, item_meta["parsing"]["columns"]
    #                       ) for item_tab_col2pos, item_meta in zip(all_tab_col2pos, all_meta_data)
    # ]

    all_val_ground_labels = [
        value_ground_sup(input_ids, item_val2pos) for input_ids, item_val2pos in zip(model_inputs["input_ids"], all_val2pos)
    ]
    all_val_ground_labels = torch.tensor(all_val_ground_labels, dtype=torch.long)
    assert all_val_ground_labels.size() == model_inputs["input_ids"].size()

    # TODO: Add value-column matching labels.

    # FIXME: Fixed 2022/07/03 1:05
    # all_tab_ground_labels[tab_mask.bool()] = -1
    # all_col_ground_labels[col_mask.bool()] = -1
    # all_val_ground_labels[model_inputs["attention_mask"].bool()] = -1
    all_tab_ground_labels[~(tab_mask.bool())] = -1
    all_col_ground_labels[~(col_mask.bool())] = -1
    all_val_ground_labels[~(model_inputs["attention_mask"].bool())] = -1

    model_inputs["tab_labels"] = all_tab_ground_labels
    model_inputs["col_labels"] = all_col_ground_labels
    model_inputs["val_labels"] = all_val_ground_labels

    return DictTensorDataset(model_inputs, all_meta_data)


def read_data_w_multi_ground_v2(file_path: str, tokenizer: PreTrainedTokenizer, db_info_path: str, dpr_results_path: str,
                                top_k_tab: int = 5, top_k_col: Union[str, int] = 5, add_tab_cn: bool = False,
                                add_col_cn: bool = False, name_lowercase: bool = False,
                                aug_with_align: bool = False, test_dpr_results: str = None, test_file: str = None,
                                add_ent_labels: bool = False, kb_file: str = None,
                                max_input_length: int = 512, max_output_length: int = 128):
    """
    Version 2 adds value-column matching supervision and lowercases the names of tables and columns.
    """

    tokenizer.add_tokens(["<table>", "<column>"])
    if add_tab_cn:
        tokenizer.add_tokens(["<cn>"])

    db_info = json.load(open(db_info_path, 'r'))

    table_names_en2col = {}
    table_names_en2col_cn = {}
    all_table_names_en = []
    all_table_names_cn = []
    for db in db_info:
        db_table_names = db["table_name"]
        if name_lowercase:
            all_table_names_en.extend([tab_name[0].lower() for tab_name in db_table_names])
        else:
            all_table_names_en.extend([tab_name[0] for tab_name in db_table_names])
        all_table_names_cn.extend([tab_name[1] for tab_name in db_table_names])
        for tab in db["column_info"]:
            tab_en_name = tab["table"]
            if name_lowercase:
                tab_col_en = [tmp.lower() for tmp in tab["columns"][2:]]  # 忽略 `*` 和 `id` 两列
            else:
                tab_col_en = tab["columns"][2:]  # 忽略 `*` 和 `id` 两列
            tab_col_cn = tab["column_chiName"][2:]

            table_names_en2col[tab_en_name] = tab_col_en
            table_names_en2col_cn[tab_en_name] = tab_col_cn

    tab_col_en2cn = get_tab_col_en2cn(db_info)

    dpr_results = json.load(open(dpr_results_path, 'r'))

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    # Hack here for back translated data !
    if len(dpr_results) < len(data):
        test_dpr_results = json.load(open(test_dpr_results, "r"))
        test_data = json.load(open(test_file, "r"))
        q_id2dpr = {}
        for dpr, item in zip(test_dpr_results, test_data):
            q_id2dpr[item["q_id"]] = dpr
        for item in data[len(dpr_results):]:
            dpr_results.append(q_id2dpr[item["q_id"]])
        assert len(dpr_results) == len(data)

    # Get english-chinese aligned examples.
    if aug_with_align:
        assert len(data) == len(dpr_results)
        q_id2dpr = {item["q_id"]: dpr for item, dpr in zip(data, dpr_results)}

        aligned_data = []
        aligned_dpr = []
        for item in data:
            res = get_aligned_examples(item, tab_col_en2cn)
            if res is not None:
                aligned_data.append(res)
                aligned_dpr.append(q_id2dpr[res["q_id"]])
        logger.info(f"Obtained {len(aligned_data)} aligned examples.")

        data.extend(aligned_data)
        dpr_results.extend(aligned_dpr)

    top_k_tab_en = []
    top_k_tab_cn = []
    top_k_col_en = []
    top_k_col_cn = []
    for item_id, item in enumerate(data):
        item_top_k_tab_en = list(map(lambda x: all_table_names_en[x], dpr_results[item_id][:top_k_tab]))
        top_k_tab_en.append(item_top_k_tab_en)

        if add_tab_cn:
            top_k_tab_cn.append(list(map(lambda x: all_table_names_cn[x], dpr_results[item_id][:top_k_tab])))
        else:
            top_k_tab_cn.append(None)

        if top_k_col == "all":
            item_top_k_col = list(map(lambda x: table_names_en2col[x], item_top_k_tab_en))
            item_top_k_col_cn = list(map(lambda x: table_names_en2col_cn[x], item_top_k_tab_en))
        else:
            item_top_k_col = list(map(lambda x: table_names_en2col[x][:top_k_col], item_top_k_tab_en))
            item_top_k_col_cn = list(map(lambda x: table_names_en2col_cn[x][:top_k_col], item_top_k_tab_en))

        top_k_col_en.append(item_top_k_col)

        if add_col_cn:
            top_k_col_cn.append(item_top_k_col_cn)
        else:
            top_k_col_cn.append(None)

    process_init(tokenizer)
    all_questions, all_tab2pos, all_tab_col2pos, all_val2pos, all_meta_data = [], [], [], [], []
    for item, item_top_k_tab_en, item_top_k_col_en, item_top_k_tab_cn, item_top_k_col_cn in tqdm(zip(data, top_k_tab_en, top_k_col_en,
                                                                                                     top_k_tab_cn, top_k_col_cn),
                                                                                                 total=len(data)):
        res = _process_single_input(item, item_top_k_tab_en, item_top_k_col_en, item_top_k_tab_cn, item_top_k_col_cn)
        all_questions.append(res[0])
        all_tab2pos.append(res[1])
        all_tab_col2pos.append(res[2])
        all_val2pos.append(res[3])
        all_meta_data.append(res[4])

    # Obtain basic labels.
    all_cls_labels = []
    all_seq_queries = []
    for item, item_meta in zip(data, all_meta_data):
        if "db_name" in item:
            item_meta["db_name"] = item["db_name"]
            all_cls_labels.append(db_vocab[item["db_name"]])

        if "sql_query" in item:
            item_meta["sql_query"] = item["sql_query"]
            all_seq_queries.append(item["sql_query"])

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

    if len(all_cls_labels):
        model_inputs["cls_labels"] = torch.tensor(all_cls_labels, dtype=torch.long)

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
        # tab_pos_index[item_id, :len(item_tab2pos)] = torch.tensor(list(item_tab2pos.values()))
        tables = [(s, e - 1) for s, e in item_tab2pos.values()]  # FIXME: Fixed at 2022/07/02
        tab_pos_index[item_id, :len(tables)] = torch.tensor(tables, dtype=torch.long)
        tab_mask[item_id, :len(item_tab2pos)] = 1

    max_column_num = 0
    for item_tab_col2pos in all_tab_col2pos:
        column_num = sum(map(len, item_tab_col2pos.values()))
        max_column_num = max(max_column_num, column_num)
    col_pos_index = torch.zeros(len(data), max_column_num, 2, dtype=torch.long)
    col_mask = torch.zeros(len(data), max_column_num, dtype=torch.long)
    for item_id, item_tab_col2pos in enumerate(all_tab_col2pos):
        columns = []
        for tab, col2pos in item_tab_col2pos.items():
            # columns.extend(col2pos.values())
            columns.extend([(s, e - 1) for s, e in col2pos.values()])  # FIXME: Fixed at 2022/07/02
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

    if add_ent_labels:
        ent_name, ent_enum = load_kb_entity_enum(kb_file)
        logger.info(f"Enum amount: {len(ent_enum)}")
        logger.info(f"Name amount: {len(ent_name)}")
        name_cnt = 0
        enum_cnt = 0
        name_labels, enum_labels = [], []
        # entity_labeling_init(tokenizer, ent_name, ent_enum)
        # for input_id in tqdm(model_inputs["input_ids"], total=model_inputs["input_ids"].size(0)):
        #     res = annotate_entity(input_id.tolist())
        #     name_labels.append(res[0])
        #     enum_labels.append(res[1])
        #     name_cnt += res[2]
        #     enum_cnt += res[3]
        input_ids = model_inputs["input_ids"].tolist()
        with Pool(8, entity_labeling_init, initargs=(tokenizer, ent_name, ent_enum)) as p:
            results = list(tqdm(
                p.imap(annotate_entity, input_ids, chunksize=32),
                total=len(input_ids)
            ))
        for res in results:
            name_labels.append(res[0])
            enum_labels.append(res[1])
            name_cnt += res[2]
            enum_cnt += res[3]

        model_inputs["name_ids"] = torch.tensor(name_labels, dtype=torch.long)
        model_inputs["enum_ids"] = torch.tensor(enum_labels, dtype=torch.long)

        logger.info(f"Annotate {name_cnt} entity attributes and {enum_cnt} entities.")

    # For debug
    num_values = val_mask.sum()
    num_val_col_mat_pairs = (all_val_col_match_labels != -1).sum()
    logger.info(f"Number of total parsed values in dataset: {num_values}.")
    logger.info(f"Number of total parsed matched value-column pairs in dataset: {num_val_col_mat_pairs}.")

    return DictTensorDataset(model_inputs, all_meta_data)


def read_data_w_multi_ground_v2_w_aug(file_path: str, tokenizer: PreTrainedTokenizer, db_info_path: str, kb_file: str,
                                      dpr_results_path: str,
                                      top_k_tab: int = 5, top_k_col: Union[str, int] = 5, add_tab_cn: bool = False,
                                      name_lowercase: bool = False, aug_tokenize: bool = False,
                                      max_input_length: int = 512, max_output_length: int = 128):
    """
    Version 2 adds value-column matching supervision and lowercases the names of tables and columns.
    """

    tokenizer.add_tokens(["<table>", "<column>"])
    if add_tab_cn:
        tokenizer.add_tokens(["<cn>"])

    db_info = json.load(open(db_info_path, 'r'))

    table_names_en2col = {}
    all_table_names_en = []
    all_table_names_cn = []
    for db in db_info:
        db_table_names = db["table_name"]
        if name_lowercase:
            all_table_names_en.extend([tab_name[0].lower() for tab_name in db_table_names])
        else:
            all_table_names_en.extend([tab_name[0] for tab_name in db_table_names])
        all_table_names_cn.extend([tab_name[1] for tab_name in db_table_names])
        for tab in db["column_info"]:
            tab_en_name = tab["table"]
            if name_lowercase:
                tab_col_en = [tmp.lower() for tmp in tab["columns"][2:]]  # 忽略 `*` 和 `id` 两列
            else:
                tab_col_en = tab["columns"][2:]  # 忽略 `*` 和 `id` 两列
            table_names_en2col[tab_en_name] = tab_col_en

    dpr_results = json.load(open(dpr_results_path, 'r'))

    logger.info("Reading knowledge base from {}".format(kb_file))
    kb = get_ent_attr_alias(kb_file)

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    data_augmentations = augmentation_with_ent_attr_alias(data, kb, tokenize=aug_tokenize)
    for aug in data_augmentations:
        orig_item_id = aug["orig_index"]
        dpr_results.append(dpr_results[orig_item_id])
    data.extend(data_augmentations)
    assert len(dpr_results) == len(data)

    top_k_tab_en = []
    top_k_tab_cn = []
    top_k_col_en = []
    for item_id, item in enumerate(data):
        item_top_k_tab_en = list(map(lambda x: all_table_names_en[x], dpr_results[item_id][:top_k_tab]))
        top_k_tab_en.append(item_top_k_tab_en)

        if add_tab_cn:
            top_k_tab_cn.append(list(map(lambda x: all_table_names_cn[x], dpr_results[item_id][:top_k_tab])))
        else:
            top_k_tab_cn.append(None)

        if top_k_col == "all":
            item_top_k_col = list(map(lambda x: table_names_en2col[x], item_top_k_tab_en))
        else:
            item_top_k_col = list(map(lambda x: table_names_en2col[x][:top_k_col], item_top_k_tab_en))
        top_k_col_en.append(item_top_k_col)

    process_init(tokenizer)
    all_questions, all_tab2pos, all_tab_col2pos, all_val2pos, all_meta_data = [], [], [], [], []
    for item, item_top_k_tab_en, item_top_k_col_en, item_top_k_tab_cn in tqdm(zip(data, top_k_tab_en, top_k_col_en, top_k_tab_cn),
                                                                              total=len(data)):
        res = _process_single_input(item, item_top_k_tab_en, item_top_k_col_en, item_top_k_tab_cn)
        all_questions.append(res[0])
        all_tab2pos.append(res[1])
        all_tab_col2pos.append(res[2])
        all_val2pos.append(res[3])
        all_meta_data.append(res[4])

    # Obtain basic labels.
    all_cls_labels = []
    all_seq_queries = []
    for item, item_meta in zip(data, all_meta_data):
        if "db_name" in item:
            item_meta["db_name"] = item["db_name"]
            all_cls_labels.append(db_vocab[item["db_name"]])

        if "sql_query" in item:
            item_meta["sql_query"] = item["sql_query"]
            all_seq_queries.append(item["sql_query"])

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

    if len(all_cls_labels):
        model_inputs["cls_labels"] = torch.tensor(all_cls_labels, dtype=torch.long)

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
        # tab_pos_index[item_id, :len(item_tab2pos)] = torch.tensor(list(item_tab2pos.values()))
        tables = [(s, e - 1) for s, e in item_tab2pos.values()]  # FIXME: Fixed at 2022/07/02
        tab_pos_index[item_id, :len(tables)] = torch.tensor(tables, dtype=torch.long)
        tab_mask[item_id, :len(item_tab2pos)] = 1

    max_column_num = 0
    for item_tab_col2pos in all_tab_col2pos:
        column_num = sum(map(len, item_tab_col2pos.values()))
        max_column_num = max(max_column_num, column_num)
    col_pos_index = torch.zeros(len(data), max_column_num, 2, dtype=torch.long)
    col_mask = torch.zeros(len(data), max_column_num, dtype=torch.long)
    for item_id, item_tab_col2pos in enumerate(all_tab_col2pos):
        columns = []
        for tab, col2pos in item_tab_col2pos.items():
            # columns.extend(col2pos.values())
            columns.extend([(s, e - 1) for s, e in col2pos.values()])  # FIXME: Fixed at 2022/07/02
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

    return DictTensorDataset(model_inputs, all_meta_data)


def read_data_w_multi_ground_v2_w_examples(file_path: str, tokenizer: PreTrainedTokenizer, db_info_path: str, dpr_results_path: str,
                                           top_k_tab: int = 5, top_k_col: Union[str, int] = 5, add_tab_cn: bool = False,
                                           add_col_cn: bool = False, name_lowercase: bool = False, train_file_path: str = None,
                                           examples_top_k: int = 3, avoid_same: bool = False,
                                           aug_with_align: bool = False,
                                           target_only: bool = False, swap: bool = False, add_original: bool = False,
                                           test_dpr_results: str = None, test_file: str = None,
                                           add_ent_labels: bool = False, kb_file: str = None,
                                           max_input_length: int = 512, max_output_length: int = 128):
    """
    Version 2 adds value-column matching supervision and lowercases the names of tables and columns.

    Add examples from training set to improve the performance.
    """

    if swap:
        tokenizer.add_tokens(["<table>", "<column>"])
    else:
        tokenizer.add_tokens(["<table>", "<column>", "<examples>", "<input>"])
        assert len(tokenizer.tokenize("<examples>")) == 1 and tokenizer.tokenize("<examples>")[0] == "<examples>"

    if add_tab_cn:
        tokenizer.add_tokens(["<cn>"])

    db_info = json.load(open(db_info_path, 'r'))

    table_names_en2col = {}
    table_names_en2col_cn = {}
    all_table_names_en = []
    all_table_names_cn = []
    for db in db_info:
        db_table_names = db["table_name"]
        if name_lowercase:
            all_table_names_en.extend([tab_name[0].lower() for tab_name in db_table_names])
        else:
            all_table_names_en.extend([tab_name[0] for tab_name in db_table_names])
        all_table_names_cn.extend([tab_name[1] for tab_name in db_table_names])
        for tab in db["column_info"]:
            tab_en_name = tab["table"]
            if name_lowercase:
                tab_col_en = [tmp.lower() for tmp in tab["columns"][2:]]  # 忽略 `*` 和 `id` 两列
            else:
                tab_col_en = tab["columns"][2:]  # 忽略 `*` 和 `id` 两列
            tab_col_cn = tab["column_chiName"][2:]

            table_names_en2col[tab_en_name] = tab_col_en
            table_names_en2col_cn[tab_en_name] = tab_col_cn

    tab_col_en2cn = get_tab_col_en2cn(db_info)

    dpr_results = json.load(open(dpr_results_path, 'r'))

    logger.info("Reading data from {}".format(file_path))
    data = json.load(open(file_path, 'r'))

    # Hack here for back translated data !
    if len(dpr_results) < len(data):
        test_dpr_results = json.load(open(test_dpr_results, "r"))
        test_data = json.load(open(test_file, "r"))
        q_id2dpr = {}
        for dpr, item in zip(test_dpr_results, test_data):
            q_id2dpr[item["q_id"]] = dpr
        for item in data[len(dpr_results):]:
            dpr_results.append(q_id2dpr[item["q_id"]])
        assert len(dpr_results) == len(data)

    # Get english-chinese aligned examples.
    if aug_with_align:
        assert len(data) == len(dpr_results)
        q_id2dpr = {item["q_id"]: dpr for item, dpr in zip(data, dpr_results)}

        aligned_data = []
        aligned_dpr = []
        for item in data:
            res = get_aligned_examples(item, tab_col_en2cn)
            if res is not None:
                aligned_data.append(res)
                aligned_dpr.append(q_id2dpr[res["q_id"]])
        logger.info(f"Obtained {len(aligned_data)} aligned examples.")

        data.extend(aligned_data)
        dpr_results.extend(aligned_dpr)

    # Obtain all query-sql pairs.
    train_data = json.load(open(train_file_path, 'r'))
    train_nl = []
    tokenized_train_nl = []
    train_sql = []
    for item in train_data:
        train_nl.append(item["question"])
        tokenized_train_nl.append(list(jieba.cut(item["question"])))
        train_sql.append(item["sql_query"])

    # Construct BM25 model
    bm25_model = BM25Model(tokenized_train_nl)

    # Obtain similar examples for each item.
    retrieved_examples = []
    for item_id, item in enumerate(data):
        question = item["question"]
        ques_words = list(jieba.cut(question))
        bm25_scores = [(idx, _score) for idx, _score in enumerate(bm25_model.get_documents_score(ques_words))]
        sorted_bm25_scores = sorted(bm25_scores, key=lambda x: x[1], reverse=True)
        _examples = []
        for _idx, _ in sorted_bm25_scores:
            if (train_file_path == file_path or avoid_same) and _idx == item_id:
                continue
            _examples.append((train_nl[_idx], train_sql[_idx]))
            if len(_examples) == examples_top_k:
                break
        retrieved_examples.append(_examples)

    # Add retrieved examples to input.
    prefixes = []
    if swap:
        for examples in retrieved_examples:
            prompt = " "
            for exp in examples:
                if target_only:
                    prompt = prompt + f"{exp[1]}"
                else:
                    prompt = prompt + f"{exp[0]} {exp[1]}"
            prefixes.append(prompt)
    else:
        for examples in retrieved_examples:
            prompt = f"<examples>"
            for exp in examples:
                if target_only:
                    prompt = prompt + f"{exp[1]}"
                else:
                    prompt = prompt + f"{exp[0]} {exp[1]}"
            # _input = f"{prompt}<input>{question}<output>"
            # _tmp.append(_input)
            # meta["debug"]["question"] = _input
            prefix = f"{prompt}<input>"
            prefixes.append(prefix)

    top_k_tab_en = []
    top_k_tab_cn = []
    top_k_col_en = []
    top_k_col_cn = []
    for item_id, item in enumerate(data):
        item_top_k_tab_en = list(map(lambda x: all_table_names_en[x], dpr_results[item_id][:top_k_tab]))
        top_k_tab_en.append(item_top_k_tab_en)

        if add_tab_cn:
            top_k_tab_cn.append(list(map(lambda x: all_table_names_cn[x], dpr_results[item_id][:top_k_tab])))
        else:
            top_k_tab_cn.append(None)

        if top_k_col == "all":
            item_top_k_col = list(map(lambda x: table_names_en2col[x], item_top_k_tab_en))
            item_top_k_col_cn = list(map(lambda x: table_names_en2col_cn[x], item_top_k_tab_en))
        else:
            item_top_k_col = list(map(lambda x: table_names_en2col[x][:top_k_col], item_top_k_tab_en))
            item_top_k_col_cn = list(map(lambda x: table_names_en2col_cn[x][:top_k_col], item_top_k_tab_en))

        top_k_col_en.append(item_top_k_col)

        if add_col_cn:
            top_k_col_cn.append(item_top_k_col_cn)
        else:
            top_k_col_cn.append(None)

    process_init(tokenizer)
    all_questions, all_tab2pos, all_tab_col2pos, all_val2pos, all_meta_data = [], [], [], [], []
    for item, item_prefix, item_top_k_tab_en, item_top_k_col_en, item_top_k_tab_cn, item_top_k_col_cn in tqdm(zip(data, prefixes,
                                                                                                                  top_k_tab_en,
                                                                                                                  top_k_col_en,
                                                                                                                  top_k_tab_cn,
                                                                                                                  top_k_col_cn),
                                                                                                              total=len(data)):
        if swap:
            res = _process_single_input(item, item_top_k_tab_en, item_top_k_col_en, item_top_k_tab_cn, item_top_k_col_cn,
                                        suffix=item_prefix)
        else:
            res = _process_single_input(item, item_top_k_tab_en, item_top_k_col_en, item_top_k_tab_cn, item_top_k_col_cn,
                                        prefix=item_prefix)
        all_questions.append(res[0])
        all_tab2pos.append(res[1])
        all_tab_col2pos.append(res[2])
        all_val2pos.append(res[3])
        all_meta_data.append(res[4])

        if add_original:
            res = _process_single_input(item, item_top_k_tab_en, item_top_k_col_en, item_top_k_tab_cn)
            all_questions.append(res[0])
            all_tab2pos.append(res[1])
            all_tab_col2pos.append(res[2])
            all_val2pos.append(res[3])
            all_meta_data.append(res[4])

    if add_original:
        ex_data = []
        for item in data:
            ex_data.extend([item] * 2)
        assert len(ex_data) == 2 * len(data)
        assert len(ex_data) == len(all_meta_data)
        data = ex_data

    # Obtain basic labels.
    all_cls_labels = []
    all_seq_queries = []
    for item, item_meta in zip(data, all_meta_data):
        if "db_name" in item:
            item_meta["db_name"] = item["db_name"]
            all_cls_labels.append(db_vocab[item["db_name"]])

        if "sql_query" in item:
            item_meta["sql_query"] = item["sql_query"]
            all_seq_queries.append(item["sql_query"])

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

    if len(all_cls_labels):
        model_inputs["cls_labels"] = torch.tensor(all_cls_labels, dtype=torch.long)

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
        # tab_pos_index[item_id, :len(item_tab2pos)] = torch.tensor(list(item_tab2pos.values()))
        tables = [(s, e - 1) for s, e in item_tab2pos.values()]
        tab_pos_index[item_id, :len(tables)] = torch.tensor(tables, dtype=torch.long)
        tab_mask[item_id, :len(item_tab2pos)] = 1

    max_column_num = 0
    for item_tab_col2pos in all_tab_col2pos:
        column_num = sum(map(len, item_tab_col2pos.values()))
        max_column_num = max(max_column_num, column_num)
    col_pos_index = torch.zeros(len(data), max_column_num, 2, dtype=torch.long)
    col_mask = torch.zeros(len(data), max_column_num, dtype=torch.long)
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

    if add_ent_labels:
        ent_name, ent_enum = load_kb_entity_enum(kb_file)
        logger.info(f"Enum amount: {len(ent_enum)}")
        logger.info(f"Name amount: {len(ent_name)}")
        name_cnt = 0
        enum_cnt = 0
        name_labels, enum_labels = [], []
        # entity_labeling_init(tokenizer, ent_name, ent_enum)
        # for input_id in tqdm(model_inputs["input_ids"], total=model_inputs["input_ids"].size(0)):
        #     res = annotate_entity(input_id.tolist())
        #     name_labels.append(res[0])
        #     enum_labels.append(res[1])
        #     name_cnt += res[2]
        #     enum_cnt += res[3]
        input_ids = model_inputs["input_ids"].tolist()
        with Pool(8, entity_labeling_init, initargs=(tokenizer, ent_name, ent_enum)) as p:
            results = list(tqdm(
                p.imap(annotate_entity, input_ids, chunksize=32),
                total=len(input_ids)
            ))
        for res in results:
            name_labels.append(res[0])
            enum_labels.append(res[1])
            name_cnt += res[2]
            enum_cnt += res[3]

        model_inputs["name_ids"] = torch.tensor(name_labels, dtype=torch.long)
        model_inputs["enum_ids"] = torch.tensor(enum_labels, dtype=torch.long)

        logger.info(f"Annotate {name_cnt} entity attributes and {enum_cnt} entities.")

    # For debug
    num_values = val_mask.sum()
    num_val_col_mat_pairs = (all_val_col_match_labels != -1).sum()
    logger.info(f"Number of total parsed values in dataset: {num_values}.")
    logger.info(f"Number of total parsed matched value-column pairs in dataset: {num_val_col_mat_pairs}.")

    return DictTensorDataset(model_inputs, all_meta_data)
