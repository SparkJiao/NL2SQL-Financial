import collections
from typing import Dict, List, Any, Union, Tuple

import torch
from torch import distributed as dist
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import jieba

from general_util.logger import get_child_logger
from general_util.sqliteStructureTrans import gen_sqlite_struct_file_from_list
from general_util import text2sql_evaluation
from general_util.mixin import DistGatherMixin
from general_util.ccks_struct_evaluation import evaluate_inline
from models.mt5 import Seq2SeqLMPredictionOutput

logger = get_child_logger("Metrics")

db_vocab = [
    "ccks_stock",
    "ccks_fund",
    "ccks_macro",
]


class SQLResultsBase:
    def __init__(self):
        self.ground_sql_query = []
        self.ground_sqlite = []
        self.pred_sql_query = []
        self.predictions = []

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Union[Seq2SeqLMPredictionOutput, Dict[str, Any]]):
        ground_sql_query = []
        ground_sqlite = []
        q_id = []
        questions = []
        for item in meta_data:
            q_id.append(item['q_id'])
            questions.append(item['question'])

            if 'sql_query' in item:
                ground_sql_query.append(item['sql_query'])
            if 'sqlite' in item:
                ground_sqlite.append(item['sqlite'])

        pred_sql_query = batch_model_outputs["generated_seq"]

        assert len(pred_sql_query) == len(ground_sql_query) or len(ground_sql_query) == 0, (len(pred_sql_query), len(ground_sql_query))

        predictions = [
            {
                "q_id": _q_id,
                "sql_query": _pred_sql_query,
                "question": _ques,
                "db_name": "",
            }
            for _q_id, _ques, _pred_sql_query in zip(q_id, questions, pred_sql_query)
        ]

        predictions = gen_sqlite_struct_file_from_list(predictions)
        assert len(predictions) == len(ground_sqlite) or len(ground_sqlite) == 0

        if "sequences_scores" in batch_model_outputs:
            for output, pred in zip(batch_model_outputs["sequences_scores"], predictions):
                pred["sequence_scores"] = output.item()

        self.ground_sql_query.extend(ground_sql_query)
        self.ground_sqlite.extend(ground_sqlite)
        self.pred_sql_query.extend(pred_sql_query)
        self.predictions.extend(predictions)

        del meta_data, batch_model_outputs
        del ground_sql_query, ground_sqlite, pred_sql_query, predictions

    @staticmethod
    def structure_equal(sqlite, pred):
        tmp = {
            "from": pred["from"],
            "select": pred["select"],
            "where": pred["where"],
            "groupBy": pred["groupBy"],
            "having": pred["having"],
            "orderBy": pred["orderBy"],
            "limit": pred["limit"]
        }
        return tmp == sqlite

    def get_results(self):
        if len(self.ground_sql_query):
            bleu = sum(
                [sentence_bleu([word_tokenize(tgt)], word_tokenize(pred)) for tgt, pred in zip(self.ground_sql_query, self.pred_sql_query)]
            ) * 1.0 / len(self.ground_sql_query)
        else:
            bleu = 0.0

        if len(self.ground_sqlite) > 0:
            em = sum(
                [1.0 if self.structure_equal(tgt, pred) else 0.0 for tgt, pred in zip(self.ground_sqlite, self.predictions)]
            ) * 1.0 / len(self.ground_sqlite)
        else:
            em = 0.0

        return {"bleu": bleu, "em": em}, self.predictions


class SQLResultsWCls(SQLResultsBase):
    def __init__(self):
        super(SQLResultsWCls, self).__init__()

        self.ground_db_names = []
        self.pred_db_names = []

    @staticmethod
    def structure_equal_w_cls(sqlite, pred):
        tmp = {
            "db_name": pred["db_name"],
            "from": pred["from"],
            "select": pred["select"],
            "where": pred["where"],
            "groupBy": pred["groupBy"],
            "having": pred["having"],
            "orderBy": pred["orderBy"],
            "limit": pred["limit"]
        }
        return tmp == sqlite

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Union[Seq2SeqLMPredictionOutput, Dict[str, Any]]):
        ground_db_names = []
        for item in meta_data:
            if "db_name" in item:
                ground_db_names.append(item["db_name"])

        cls_logits = batch_model_outputs["cls_logits"].detach().float().cpu()
        _, cls_pred = cls_logits.max(dim=-1)
        pred_db_names = [db_vocab[_cls_pred.item()] for _cls_pred in cls_pred]

        self.ground_db_names.extend(ground_db_names)
        self.pred_db_names.extend(pred_db_names)
        assert len(self.ground_db_names) == len(self.pred_db_names) or len(self.ground_db_names) == 0, (
            len(self.ground_db_names), len(self.pred_db_names))

        super().__call__(meta_data, batch_model_outputs)

        assert len(self.predictions) == len(self.pred_db_names), (len(self.predictions), len(self.pred_db_names))

    def get_results(self):
        results, _ = super().get_results()

        assert len(self.ground_sqlite) == len(self.ground_db_names), (len(self.ground_sqlite), len(self.ground_db_names))
        for sqlite, ground_db_name in zip(self.ground_sqlite, self.ground_db_names):
            sqlite["db_name"] = ground_db_name

        assert len(self.pred_db_names) == len(self.predictions), (len(self.pred_db_names), len(self.predictions))
        for pred_db_name, prediction in zip(self.pred_db_names, self.predictions):
            prediction["db_name"] = pred_db_name

        if len(self.ground_sqlite):
            cls_em = sum(
                [1.0 if self.structure_equal_w_cls(tgt, pred) else 0.0 for tgt, pred in zip(self.ground_sqlite, self.predictions)]
            ) * 1.0 / len(self.ground_sqlite)
        else:
            cls_em = 0.0

        results["cls_em"] = cls_em

        return results, self.predictions


class SQLResultsWClsHelper(SQLResultsBase):
    def __init__(self, official_evaluate: bool = False):
        super(SQLResultsWClsHelper, self).__init__()

        self.official_evaluate = official_evaluate

        self.ground_db_names = []
        self.pred_db_names = []
        self.meta_data = []

    @staticmethod
    def structure_equal_w_cls(sqlite, pred):
        tmp = {
            "db_name": pred["db_name"],
            "from": pred["from"],
            "select": pred["select"],
            "where": pred["where"],
            "groupBy": pred["groupBy"],
            "having": pred["having"],
            "orderBy": pred["orderBy"],
            "limit": pred["limit"]
        }
        return tmp == sqlite

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Union[Seq2SeqLMPredictionOutput, Dict[str, Any]]):
        ground_db_names = []
        for item in meta_data:
            if "db_name" in item:
                ground_db_names.append(item["db_name"])
            self.meta_data.append({"parsing": item["parsing"], "debug": item["debug"]})

        cls_logits = batch_model_outputs["cls_logits"].detach().float().cpu()
        _, cls_pred = cls_logits.max(dim=-1)
        pred_db_names = [db_vocab[_cls_pred.item()] for _cls_pred in cls_pred]

        self.ground_db_names.extend(ground_db_names)
        self.pred_db_names.extend(pred_db_names)
        assert len(self.ground_db_names) == len(self.pred_db_names) or len(self.ground_db_names) == 0, (
            len(self.ground_db_names), len(self.pred_db_names))

        super().__call__(meta_data, batch_model_outputs)

        assert len(self.predictions) == len(self.pred_db_names), (len(self.predictions), len(self.pred_db_names))

    def get_results(self):
        results, _ = super().get_results()

        assert len(self.ground_sqlite) == len(self.ground_db_names), (len(self.ground_sqlite), len(self.ground_db_names))
        for sqlite, ground_db_name in zip(self.ground_sqlite, self.ground_db_names):
            sqlite["db_name"] = ground_db_name

        assert len(self.pred_db_names) == len(self.predictions), (len(self.pred_db_names), len(self.predictions))
        for pred_db_name, prediction in zip(self.pred_db_names, self.predictions):
            prediction["db_name"] = pred_db_name

        if len(self.ground_sqlite):
            if self.official_evaluate:
                for pred, sqlite in zip(self.predictions, self.ground_sqlite):
                    sqlite["q_id"] = pred["q_id"]
                cls_em = evaluate_inline(self.predictions, self.ground_sqlite)
            else:
                cls_em = sum(
                    [1.0 if self.structure_equal_w_cls(tgt, pred) else 0.0 for tgt, pred in zip(self.ground_sqlite, self.predictions)]
                ) * 1.0 / len(self.ground_sqlite)
        else:
            cls_em = 0.0

        results["cls_em"] = cls_em

        for prediction, meta in zip(self.predictions, self.meta_data):
            prediction["meta"] = meta

        return results, self.predictions


class DuSQLResultsHelper(DistGatherMixin):
    def __init__(self, db_schema_path: str):
        super(DuSQLResultsHelper, self).__init__()

        self.db_schema_path = db_schema_path

        self.predictions = []
        self.golds = []
        self.meta_data = []

    def __call__(self, meta_data: List[Dict[str, Any]],
                 batch_model_outputs: Union[Seq2SeqLMPredictionOutput, Dict[str, Any]],
                 ddp: bool = False):
        # Data format: qid\tsql_query\tdb_id

        pred_sql_query = batch_model_outputs["generated_seq"]
        golds = []
        predictions = []
        metas = []
        for item, pred in zip(meta_data, pred_sql_query):
            if "sql_query" in item:
                golds.append({
                    "q_id": item["q_id"],
                    "sql_query": item["sql_query"],
                    "db_id": item["db_name"]
                })
            predictions.append({
                "q_id": item["q_id"],
                "sql_query": pred,
                "db_id": item["db_name"]
            })
            metas.append({
                "parsing": item["parsing"],
                "question": item["question"],
                "db_name": item["db_name"],
            })

        if ddp:
            obj = list(zip(golds, predictions, metas))
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                tmp_a, tmp_b, tmp_c = [], [], []
                for item in gather_res:
                    tmp_a.extend(list(map(lambda x: x[0], item)))
                    tmp_b.extend(list(map(lambda x: x[1], item)))
                    tmp_c.extend(list(map(lambda x: x[2], item)))
                golds = tmp_a
                predictions = tmp_b
                metas = tmp_c

        self.predictions.extend(predictions)
        self.golds.extend(golds)
        self.meta_data.extend(metas)

    def get_results(self):
        gold_dict = {
            item["q_id"]: [item["sql_query"], item["db_id"]] for item in self.golds
        }
        pred_dict = {
            item["q_id"]: [item["sql_query"], item["db_id"]] for item in self.predictions
        }

        scores, _ = text2sql_evaluation.evaluate_complex_readin(self.db_schema_path, gold_dict, pred_dict, mode='exact', single_equal=True)

        logger.info(f"Full metrics: {scores}")

        for pred, meta in zip(self.predictions, self.meta_data):
            pred["question"] = meta["question"]
            pred["db_name"] = meta["db_name"]
            pred["parsing"] = meta["parsing"]

        return scores["all"], self.predictions


def sql_get_results_base(meta_data: List[Dict[str, Any]], model_outputs: List[Union[Seq2SeqLMPredictionOutput, Dict[str, Any]]]):
    ground_sql_query = []
    ground_sqlite = []
    q_id = []
    questions = []
    for item in meta_data:
        q_id.append(item['q_id'])
        questions.append(item['question'])

        if 'sql_query' in item:
            ground_sql_query.append(item['sql_query'])
        if 'sqlite' in item:
            ground_sqlite.append(item['sqlite'])

    pred_sql_query = []
    for batch_model_output in model_outputs:
        pred_sql_query.extend(batch_model_output["generated_seq"])

    assert len(pred_sql_query) == len(ground_sql_query) or len(ground_sql_query) == 0

    predictions = [
        {
            "q_id": _q_id,
            "sql_query": _pred_sql_query,
            "question": _ques,
            "db_name": "",
        }
        for _q_id, _ques, _pred_sql_query in zip(q_id, questions, pred_sql_query)
    ]

    sqlite_struct_list = gen_sqlite_struct_file_from_list(predictions)
    assert len(sqlite_struct_list) == len(ground_sqlite) or len(ground_sqlite) == 0

    if len(ground_sql_query):
        bleu = sum(
            [sentence_bleu([word_tokenize(tgt)], word_tokenize(pred)) for tgt, pred in zip(ground_sql_query, pred_sql_query)]
        ) * 1.0 / len(ground_sql_query)
    else:
        bleu = 0.0

    if len(ground_sqlite) > 0:
        em = sum([1.0 if tgt == pred else 0.0 for tgt, pred in zip(ground_sqlite, sqlite_struct_list)]) * 1.0 / len(ground_sqlite)
    else:
        em = 0.0

    del model_outputs

    return {"bleu": bleu, "em": em}, sqlite_struct_list


def sql_get_results_w_cls(meta_data: List[Dict[str, Any]], model_outputs: List[Union[Seq2SeqLMPredictionOutput, Dict[str, Any]]]):
    cls_logits = [output["cls_logits"].detach().cpu() for output in model_outputs]
    cls_logits = torch.cat(cls_logits, dim=0)
    _, cls_pred = cls_logits.max(dim=-1)
    pred_db_names = [db_vocab[_cls_pred.item()] for _cls_pred in cls_pred]

    results, sqlite_struct_list = sql_get_results_base(meta_data, model_outputs)
    assert len(sqlite_struct_list) == len(pred_db_names)

    for item, db_name in zip(sqlite_struct_list, pred_db_names):
        item['db_name'] = db_name

    return results, sqlite_struct_list


class RetrievalResultsBase:
    def __init__(self, top: Union[List[int], Tuple[int]] = (1, 3, 5, 10)):
        self.scores = []
        self.answer_ids: List[List[int]] = []
        self.top = top

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any]):
        self.scores.append(batch_model_outputs["scores"])
        self.answer_ids.extend([meta["answer_id"] for meta in meta_data])
        del batch_model_outputs, meta_data

    def get_results(self):
        scores = torch.cat(self.scores, dim=0)
        _, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        logger.info(sorted_indices.size())
        sorted_indices = sorted_indices.tolist()

        recall = collections.defaultdict(list)
        for answer_id, predictions in zip(self.answer_ids, sorted_indices):
            for k in self.top:
                recall_k = len(set(answer_id) & set(predictions[:k])) * 1.0 / len(answer_id)
                recall["recall@{}".format(k)].append(recall_k)

        res = {}
        for k in recall:
            res[k] = sum(recall[k]) * 1.0 / len(recall[k]) if len(recall[k]) > 0 else 0.0

        return res, sorted_indices


class SQL2NL:
    def __init__(self):
        self.ground_question = []
        self.pred_question = []
        self.predictions = []

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Union[Seq2SeqLMPredictionOutput, Dict[str, Any]]):
        gd_question = []
        q_id = []
        for item in meta_data:
            q_id.append(item["q_id"])
            gd_question.append(item["question"])

        pd_question = batch_model_outputs["generated_seq"]
        predictions = [
            {
                "q_id": _q_id,
                "question": _ques,
                "pred_question": _pred_ques,
            } for _q_id, _ques, _pred_ques in zip(q_id, gd_question, pd_question)
        ]

        self.ground_question.extend(gd_question)
        self.pred_question.extend(pd_question)
        self.predictions.extend(predictions)

        del meta_data, batch_model_outputs, gd_question, pd_question, q_id, predictions

    def get_results(self):
        # bleu = sum(
        #     [sentence_bleu([word_tokenize(tgt)], word_tokenize(pred)) for tgt, pred in zip(self.ground_question, self.pred_question)]
        # ) * 1.0 / len(self.ground_question)
        bleu = sum(
            [sentence_bleu([list(jieba.cut(tgt))], list(jieba.cut(pred))) for tgt, pred in zip(self.ground_question, self.pred_question)]
        ) * 1.0 / len(self.ground_question)

        return {"bleu": bleu}, self.predictions
