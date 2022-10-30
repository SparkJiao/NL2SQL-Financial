#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#   Copyright (c) 2022 Hundsun Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
评估脚本：计算经sqliteStructureTrans.py脚本转换后的预测SQL结构体与真实SQL结构体匹配ACCURACY
推荐使用python3.6环境

结构体描述:
    [
      {
        "q_id": 0,
        "question": "统计不同类型基金产品的日回报率最大值",
        "db_name": "ccks_fund",
        "sql_query": "SELECT b.FundType, MAX(a.NVDailyGrowthRate) FROM mf_netvalueperformancehis AS a JOIN mf_fundarchives AS b ON a.InnerCode = b.InnerCode GROUP BY b.FundType;",
        "from": {
          "table_units": [
            "mf_netvalueperformancehis",   // 表1
            "mf_fundarchives"  // 表2
          ],
          "conds": [
            [
              "mf_netvalueperformancehis",
              "InnerCode"   // 关联字段名
            ],
            [
              "mf_fundarchives",
              "InnerCode"  // 关联字段名
            ]
          ]
        },
        "select": [
          false,  // 是否distinct
          [
            null,   // agg操作符
            "mf_fundarchives",  // 表名
            "FundType",   // 字段名
            false   // 是否distinct
          ],
          [
            "max",
            "mf_netvalueperformancehis",
            "NVDailyGrowthRate",
            false
          ]
        ],
        "where": [],  // [[[tab_name, col_name, where_op, value, datetime_op], and, []], and, [[]]]
        "groupBy": [
          [
            "mf_fundarchives",  // 表名
            "FundType"    // 字段名
          ]
        ],
        "having": [],   // [[agg_op, is_distinct, tab_name, col_name, where_op, value],..]
        "orderBy": [],   // [[desc/asc/null, agg_op, is_distinct, tab_name, col_name], ...]
        "limit": null   // null或者数值
      }
    ]

notes:
    1. having clause 下必有aggregating operation
    2. 对于from、where、having、select等部分的多个成分，如果仅是顺序不同，也算对
"""

import json
import copy
from argparse import ArgumentParser
from collections import Counter


def load_json(fpath: str) -> list:
    """
    Args: File Path
    Returns: data
    """
    with open(fpath, 'r', encoding='utf8') as fl:
        data = json.load(fl)
    return data


def get_scores_disorder_hashed(p_cont: list, g_cont: list) -> bool:
    def process(item: str) -> str:
        return item.strip().lower()

    p_cont = list(map(process, p_cont))
    g_cont = list(map(process, g_cont))
    return Counter(p_cont) == Counter(g_cont)


def get_scores_disorder(p_in: list, g_in: list) -> bool:
    g_cont = copy.deepcopy(g_in)
    p_cont = copy.deepcopy(p_in)
    while g_cont:
        cont = g_cont.pop(0)
        if cont in p_cont:
            p_cont.remove(cont)
        else:
            return False
    if not p_cont:
        return True
    else:
        return False


def get_group_counter(group: list, table_pos_index=None, p_type='nonhaving', distinct=False) -> list:
    """
    Args:
    Returns:
    """
    group_counter = []
    if p_type == 'having':
        for sub_group in group:
            if table_pos_index:
                sub_group[table_pos_index] = None
            sub_group[5] = str(sub_group[5]).strip('\'\"')
            group_counter.append("__".join([str(tmp).strip() for tmp in sub_group]).lower())
        return group_counter
    for idx, sub_group in enumerate(group):
        if table_pos_index:
            sub_group[table_pos_index] = None
        temp = "__".join([str(tmp).strip() for tmp in sub_group]).lower()
        if idx == 0 and distinct:
            group_counter.append(str(distinct).strip().lower() + '__' + temp)
        else:
            # group_counter.append(str(sub_group).lower())
            group_counter.append(temp)
    return group_counter


def get_from_group(group: list) -> list:
    """
    Args:
    Returns:
    """

    def process(obj):
        if isinstance(obj, list):
            return '__'.join([str(tmp).strip().lower() for tmp in obj])
        else:
            return str(obj).strip().lower()

    group_counter = []
    for sub_group in group:
        # tmp_group = list(map(lambda x: str(x).lower(), sub_group))
        tmp_group = list(map(process, sub_group))
        group_counter.append(Counter(tmp_group))
    return group_counter


def check_from_clause(p_from: dict, g_from: dict) -> int:
    """
    Args:
    Returns:
    """
    is_units_true = get_scores_disorder_hashed(p_from['table_units'], g_from['table_units'])
    if len(p_from['conds']) == len(g_from['conds']):
        p_group_counter = get_from_group(p_from['conds'])
        g_group_counter = get_from_group(g_from['conds'])
        is_conds_true = get_scores_disorder(p_group_counter, g_group_counter)
    else:
        is_conds_true = 0
    return int(is_units_true and is_conds_true)


def deal(list_ori: list, p: str) -> list:
    list_new = []  # 处理后的列表，是一个二维列表
    list_short = []  # 用于存放每一段列表
    for i in list_ori:
        if i != p:
            list_short.append(i)
        else:
            list_new.append(list_short)
            list_short = []
    list_new.append(list_short)  # 最后一段遇不到切割标识，需要手动放入
    return list_new


def get_link_group(group: list, level: int, table_idx: int, val_idx: int, link: str = 'or') -> str:
    if not group:
        return str(group)
    if type(group) != list:
        return str(group).strip().lower()
    while 1:
        if len(group) == 1 and type(group[0]) == list:
            group = group[0]
        elif not group:
            return str(group)
        else:
            break
    if all(type(e) != list for e in group):
        group[table_idx] = None
        # 排除"/'的影响
        group[val_idx] = str(group[val_idx]).strip(' \'\"')
        # 排除空格的影响
        return '+'.join(list(map(lambda x: str(x), group))).lower().replace(' ', '') + "+level" + str(level)
    if any(type(e) == list for e in group) and (link in group):
        group = deal(group, link)
        group = [get_link_group(tmp, level + 1, table_idx, val_idx, link=link) for tmp in group]
        group.sort()
        return "+".join(group) + "+level" + str(level)
    else:
        for idx, sub_group in enumerate(group):
            tmp_group = get_link_group(sub_group, level + 1, table_idx, val_idx, link=link)
            group[idx] = tmp_group
        group.sort()
        return "+".join(group) + "+level" + str(level)


def check_where_clause(p_where: list, g_where: list) -> bool:
    """
    Args:
    Returns:
    条件1 and 条件2  or 条件3 and 条件4   注意and的优先级高于or，即先and再or
    """
    return get_link_group(p_where, 0, 0, 3) == get_link_group(g_where, 0, 0, 3)


def check_select_clause(p_select: list, g_select: list) -> bool:
    """
    Args:
        p_select: select子句结构体预测结果
        g_select: select子句结构体真实结果
    Returns: boolean
    """
    if p_select == g_select:
        return True
    if not (p_select and g_select):
        return False
    if str(p_select[0]).strip().lower() == str(g_select[0]).strip().lower():
        p_group_counter = get_group_counter(p_select[1:], table_pos_index=1, distinct=p_select[0])
        g_group_counter = get_group_counter(g_select[1:], table_pos_index=1, distinct=g_select[0])
        return Counter(p_group_counter) == Counter(g_group_counter)
    else:
        return False


def element_remove(group: list, index: int):
    if group:
        new_group = []
        for sub_group in group:
            new_group.append('__'.join([str(item).strip() if idx != index else str(None) for idx, item in enumerate(sub_group)]).lower())
        return new_group
    else:
        return group


def check_having_clause(p_having: list, g_having: list) -> bool:
    """
    Args:
    Returns:
    只有一层
    """
    # if len(p_having) != len(g_having):
    #     return False
    # else:
    #     p_having_new = get_group_counter(p_having, 2, 'having')
    #     g_having_new = get_group_counter(g_having, 2, 'having')
    #     return Counter(g_having_new) == Counter(p_having_new)
    return get_link_group(p_having, 0, 2, 5) == get_link_group(g_having, 0, 2, 5)


def check_groupBy_clause(p_groupBy: list, g_groupBy: list) -> bool:
    """
    Args:
    Returns:
    groupBy 也有序
    """
    p_groupBy_new = element_remove(p_groupBy, 0)
    g_groupBy_new = element_remove(g_groupBy, 0)
    return p_groupBy_new == g_groupBy_new


def check_limit_clause(p_limit: int or None, g_limit: int or None) -> bool:
    """
    Args:
    Returns:
    """
    return str(p_limit).lower() == str(g_limit).lower()


def check_orderBy_clause(p_orderBy: list, g_orderBy: list) -> bool:
    """
    Args:
    Returns:
    order比较有序
    """
    p_orderBy_new = element_remove(p_orderBy, 3)
    g_orderBy_new = element_remove(g_orderBy, 3)
    return p_orderBy_new == g_orderBy_new


def evaluate(predict: str, gold: str, l: str) -> int:
    glist = load_json(gold)
    plist = load_json(predict)
    glist.sort(key=lambda x: x['q_id'])
    plist.sort(key=lambda x: x['q_id'])
    status_record = [0] * len(glist)
    # 解析sql结构体对
    partial_acc_count = 0
    for kkk, g in enumerate(glist):
        ps = [x for x in plist if x['q_id'] == g['q_id']]
        if len(ps) == 0:
            continue
        elif len(ps) > 1:
            continue
        else:
            p = ps[0]
        # check db_name库名，对加一，不对直接跳过
        if p['db_name'].strip().lower() != g['db_name'].lower():
            continue
        is_sel_true = check_select_clause(p['select'], g['select'])
        if not is_sel_true:
            continue
        is_from_true = check_from_clause(p['from'], g['from'])
        if not is_from_true:
            continue
        is_where_true = check_where_clause(p['where'], g['where'])
        if not is_where_true:
            continue
        is_grpby_true = check_groupBy_clause(p['groupBy'], g['groupBy'])
        if not is_grpby_true:
            continue
        is_order_true = check_orderBy_clause(p['orderBy'], g['orderBy'])
        if not is_order_true:
            continue
        is_hav_true = check_having_clause(p['having'], g['having'])
        if not is_hav_true:
            continue
        is_limit_true = check_limit_clause(p['limit'], g['limit'])
        if is_limit_true:
            partial_acc_count += 1
        status_record[kkk] = 1
    partial_acc = partial_acc_count / len(glist)
    with open(l, "w", encoding="utf-8") as fw:
        fw.writelines(str(partial_acc) + "###submision success")
    # print(partial_acc)
    # return partial_acc, status_record
    return 0


def evaluate_inline(plist: list, glist: list) -> float:
    # glist = load_json(gold)
    # plist = load_json(predict)
    glist.sort(key=lambda x: x['q_id'])
    plist.sort(key=lambda x: x['q_id'])
    status_record = [0] * len(glist)
    # 解析sql结构体对
    partial_acc_count = 0
    for kkk, g in enumerate(glist):
        ps = [x for x in plist if x['q_id'] == g['q_id']]
        if len(ps) == 0:
            continue
        elif len(ps) > 1:
            continue
        else:
            p = ps[0]
        # check db_name库名，对加一，不对直接跳过
        if p['db_name'].strip().lower() != g['db_name'].lower():
            continue
        is_sel_true = check_select_clause(p['select'], g['select'])
        if not is_sel_true:
            continue
        is_from_true = check_from_clause(p['from'], g['from'])
        if not is_from_true:
            continue
        is_where_true = check_where_clause(p['where'], g['where'])
        if not is_where_true:
            continue
        is_grpby_true = check_groupBy_clause(p['groupBy'], g['groupBy'])
        if not is_grpby_true:
            continue
        is_order_true = check_orderBy_clause(p['orderBy'], g['orderBy'])
        if not is_order_true:
            continue
        is_hav_true = check_having_clause(p['having'], g['having'])
        if not is_hav_true:
            continue
        is_limit_true = check_limit_clause(p['limit'], g['limit'])
        if is_limit_true:
            partial_acc_count += 1
        status_record[kkk] = 1
    partial_acc = partial_acc_count * 1.0 / len(glist)
    # with open(l, "w", encoding="utf-8") as fw:
    #     fw.writelines(str(partial_acc) + "###submision success")
    # print(partial_acc)
    # return partial_acc, status_record
    return partial_acc


# arg_parser = ArgumentParser(description='Test for argparse')
# arg_parser.add_argument('-hp', help='学生提交文件')
# arg_parser.add_argument('-rf', help='答案文件')
# arg_parser.add_argument('-l', help='结果文件')
# args = arg_parser.parse_args()
#
# if __name__ == "__main__":
#
#     try:
#         evaluate(args.hp, args.rf, args.l)
#         # evaluate(
#         #     "./test/train_test0613.json",
#         #     "./test/train.json",
#         #     "./output.log")
#     except Exception as e:
#         with open(args.l, "w", encoding="utf-8") as f:
#             f.write("提交文件格式不正确，建议基于官方提供的sqliteStructureTrans.py脚本自动转换，错误原因为： " + str(e))
