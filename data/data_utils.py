import collections

from transformers import PreTrainedTokenizer
from general_util.metrics import gen_sqlite_struct_file_from_list
from typing import Dict, List, Set, Tuple


def tokenizer_get_name(_tokenizer: PreTrainedTokenizer):
    tokenizer_name = _tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()
    return tokenizer_name


def sql_query_filter(data):
    """ Filter out the SQL queries not correctly annotated. """
    outputs = []
    for item in data:
        sql_query = item["sql_query"]
        pseudo_pred = {
            "q_id": item["q_id"],
            "sql_query": sql_query,
            "question": item["question"],
            "db_name": item["db_name"]
        }
        pseudo_structure = gen_sqlite_struct_file_from_list([pseudo_pred])[0]
        pseudo_structure = {
            "from": pseudo_structure["from"],
            "select": pseudo_structure["select"],
            "where": pseudo_structure["where"],
            "groupBy": pseudo_structure["groupBy"],
            "having": pseudo_structure["having"],
            "orderBy": pseudo_structure["orderBy"],
            "limit": pseudo_structure["limit"]
        }

        gt = {
            "from": item["from"],
            "select": item["select"],
            "where": item["where"],
            "groupBy": item["groupBy"],
            "having": item["having"],
            "orderBy": item["orderBy"],
            "limit": item["limit"]
        }

        if pseudo_structure == gt:
            outputs.append(item)

    return outputs


"""
Parsing the tables, columns, entities (values) from the SQLite structure.
The rules are referenced from the annotation in https://www.biendata.xyz/competition/fin_nl2sql/evaluation/
"""


def parse_from_from(structure: Dict):
    # TODO: 不确定``distinct``关键字有什么用，目前直接被忽略了
    _from = structure["from"]
    tables = _from["table_units"]
    _conds = _from["conds"]  # `sqliteStructureTrans.py`:Line#131: ``conds.append([[tab1, col1], [tab2, col2]])``
    columns = collections.defaultdict(list)
    for _cond in _conds:
        tup1, tup2 = _cond
        tab1, col1 = tup1
        tab2, col2 = tup2
        assert tab1 in tables and tab2 in tables, (tables, tab1, tab2)
        columns[tab1].append(col1)
        columns[tab2].append(col2)
    return tables, columns


def parse_from_select(structure: Dict):
    _select = structure["select"]
    _distinct = _select[0]
    tables = []
    columns = collections.defaultdict(list)
    for select_info in _select[1:]:  # `sqliteStructureTrans.py`:Line#208: ``sel_info.append([agg_op, tab_name, col_name, is_distinct_i])``
        agg_op, tab_name, col_name, is_distinct_i = select_info
        if tab_name is None:
            tab_name = "$TEMP"
        tables.append(tab_name)  # ``tab_name`` can be ``null`` if there is exactly one table involved.
        columns[tab_name].append(col_name)
    return tables, columns


def recursive_parse_where_tuple(where_tuple, tables, columns, values):
    if len(where_tuple) == 3:
        left, s, right = where_tuple
        assert s == "and", where_tuple
        recursive_parse_where_tuple(left, tables, columns, values)
        recursive_parse_where_tuple(right, tables, columns, values)
    elif len(where_tuple) == 1:
        recursive_parse_where_tuple(where_tuple[0], tables, columns, values)
    elif len(where_tuple) == 5:
        tab_name, col_name, where_op, val_tmp, date_op = where_tuple
        if tab_name is None:
            tab_name = "$TEMP"
        assert isinstance(tab_name, str), tab_name
        assert isinstance(col_name, str), col_name
        tables.append(tab_name)
        columns[tab_name].append(col_name)
        values.append((val_tmp, tab_name, col_name))
    elif len(where_tuple) == 0:
        return
    else:
        raise ValueError(where_tuple)


def parse_from_where(structure: Dict):
    _where = structure["where"]
    tables = []
    columns = collections.defaultdict(list)
    values = []
    # recursive_parse_where_tuple(_where, tables, columns, values)
    for where_tuple in _where:
        #``sqliteStructureTrans.py`: Line#338: ``one_constraint = [tab_name, col_name, where_op, val_tmp, date_op]``
        if isinstance(where_tuple, list):
            tab_name, col_name, where_op, val_tmp, date_op = where_tuple[0]
            if tab_name is None:
                tab_name = "$TEMP"
            tables.append(tab_name)
            columns[tab_name].append(col_name)
            values.append((val_tmp, tab_name, col_name))
        elif where_tuple == "and" or where_tuple == "or":
            continue
        else:
            raise ValueError(where_tuple)
    # if isinstance(where_tuple, list):
    #     if len(where_tuple) ==
    return tables, columns, values


def parse_from_group_by(structure: Dict):
    group_by = structure["groupBy"]
    tables = []
    columns = collections.defaultdict(list)
    for group_by_tuple in group_by:
        tab_name, col_name = group_by_tuple
        if tab_name is None:
            tab_name = "$TEMP"
        tables.append(tab_name)
        columns[tab_name].append(col_name)

    return tables, columns


def parse_from_having(structure: Dict):
    having = structure["having"]
    tables = []
    columns = collections.defaultdict()
    # TODO: Difficult to parse the table name and column names. The examples are too few.
    pass


def parse_from_limit(structure: Dict):
    # No need to implement? The value of ``limit`` keyword is a mapping from natural language to number.
    pass


def union_grounding(all_tables: Set, all_columns: Dict[str, Set], all_values: List[Tuple[str, str, str]],
                    tables: List, columns: Dict[str, List], values: List[Tuple[str, str, str]] = None):
    all_tables.update(tables)
    if "$TEMP" in all_tables:
        all_tables.remove("$TEMP")
    for tab, col_ls in columns.items():
        # if tab == "$TEMP":
        #     # assert len(all_tables) == 1, all_tables  # FIXME: 这里有问题，不知道怎么处理没给出tab的情况下且表的数量不唯一的情况
        #     tab = list(all_tables)[0]
        if tab == "$TEMP" and len(all_tables) == 1:
            tab = list(all_tables)[0]
        all_columns[tab].update(col_ls)

    if values is not None:
        for val in values:
            if val[1] == "$TEMP" and len(all_tables) == 1:
                val = (val[0], list(all_tables)[0], val[2])
            if val not in all_values:
                all_values.append(val)

    return all_tables, all_columns, all_values


def parse_grounding_from_structure(structure: Dict):
    tables = set()
    columns = collections.defaultdict(set)
    values = []

    from_tab, from_column = parse_from_from(structure)
    union_grounding(tables, columns, values, from_tab, from_column)

    select_tab, select_column = parse_from_select(structure)
    union_grounding(tables, columns, values, select_tab, select_column)

    where_tab, where_column, where_value = parse_from_where(structure)
    union_grounding(tables, columns, values, where_tab, where_column, where_value)

    group_by_tab, group_by_column = parse_from_group_by(structure)
    union_grounding(tables, columns, values, group_by_tab, group_by_column)

    columns_dict_list = {}
    for tab, col_set in columns.items():
        columns_dict_list[tab] = list(col_set)

    return list(tables), columns_dict_list, values
