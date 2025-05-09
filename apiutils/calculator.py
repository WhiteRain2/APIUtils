# SPDX-License-Identifier: MIT

"""
Module: Calculator

提供信息检索和生成模型输出的评估指标计算，包括 MRR、BLEU、MAP、Success@1、
Precision@k、Recall@k、NDCG@k 以及批量多 k 值一次性计算。
"""

import math
from functools import cached_property
from typing import List, Sequence, NamedTuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class MetricsResult(NamedTuple):
    mrr:                float
    bleu:               float
    map:                float
    successrate_at_ks:  List[float]
    precision_at_ks:    List[float]
    recall_at_ks:       List[float]
    ndcg_at_ks:         List[float]


class Calculator:
    """
    计算多对序列和答案列表的各种检索与生成质量指标。

    Attributes:
        seq_lists (List[List[str]]): 候选序列列表，每个子列表为一组 API 列表。
        answer_lists (List[List[str]]): 真实答案列表，每个子列表为对应的正确 API 列表。
        num_pairs (int): 成对序列数量。
        relevance (List[List[int]]): 每个 API 与答案的相关性评分（2=完全匹配，1=前缀匹配，0=不匹配）。
    """

    def __init__(
        self,
        seq_lists: Sequence[Sequence[str]],
        answer_lists: Sequence[Sequence[str]]
    ) -> None:
        """
        初始化 Calculator 并检查输入一致性。

        Args:
            seq_lists: 候选序列二维列表。
            answer_lists: 真实答案二维列表，长度需与 seq_lists 相同。

        Raises:
            ValueError: 当两者长度不一致或存在重复元素时抛出。
        """
        # 检查输入长度一致
        if len(seq_lists) != len(answer_lists):
            raise ValueError("seq_lists and answer_lists must have the same length")
        # 检查内部无重复
        for seq, ans in zip(seq_lists, answer_lists):
            if len(seq) != len(set(seq)):
                raise ValueError("seq_lists contains duplicate elements")
            if len(ans) != len(set(ans)):
                raise ValueError("answer_lists contains duplicate elements")

        self.seq_lists: Sequence[Sequence[str]] = seq_lists
        self.answer_lists: Sequence[Sequence[str]] = answer_lists
        self.num_pairs: int = len(seq_lists)
        # 计算相关性矩阵
        self.relevance: List[List[int]] = self.compute_relevance()

    def compute_relevance(self) -> List[List[int]]:
        """
        为每对序列和答案计算相关性评分。

        Returns:
            List[List[int]]: 与 seq_lists 对应的相关性评分列表。
        """
        relevance: List[List[int]] = []
        for seq, ans in zip(self.seq_lists, self.answer_lists):
            rel: List[int] = []
            for api in seq:
                # 完全匹配得分 2
                if api in ans:
                    rel.append(2)
                # 类匹配得分 1，用于NDCG的计算
                elif any(api.split('.')[:-1] == a.split('.')[:-1] for a in ans):
                    rel.append(1)
                else:
                    rel.append(0)
            relevance.append(rel)
        return relevance

    @cached_property
    def mrr(self) -> float:
        """
        计算所有序列对的 MRR（Mean Reciprocal Rank）平均值。

        Returns:
            float: MRR 平均值。
        """
        mrr_values: List[float] = []
        # 对每组相关性取第一个完全匹配的倒数
        for rel in self.relevance:
            ranks = [i + 1 for i, r in enumerate(rel) if r == 2]
            mrr = 1.0 / ranks[0] if ranks else 0.0
            mrr_values.append(mrr)
        return sum(mrr_values) / self.num_pairs if self.num_pairs else 0.0

    @cached_property
    def bleu(self) -> float:
        """
        计算所有序列对的平均 BLEU 值。

        Returns:
            float: BLEU 平均值。
        """
        bleu_values: List[float] = []
        smoothie = SmoothingFunction().method2
        for seq, ans in zip(self.seq_lists, self.answer_lists):
            if not seq:
                bleu_values.append(0.0)
            else:
                # 根据序列长度调整权重
                max_n = min(len(seq), len(ans), 4)
                if max_n == 0:
                    bleu_values.append(0.0)
                    continue

                weights = tuple([1.0/max_n] * max_n + [0.0] * (4-max_n))
                bleu_val = sentence_bleu([ans], seq, weights=weights, smoothing_function=smoothie)
                bleu_values.append(bleu_val)
        return sum(bleu_values) / self.num_pairs if self.num_pairs else 0.0

    @cached_property
    def map(self) -> float:
        """
        计算所有序列对的 MAP（Mean Average Precision）平均值，并缓存结果。

        Returns:
            float: MAP 平均值。
        """
        map_values: List[float] = []
        for rel, ans in zip(self.relevance, self.answer_lists):
            r = len(ans)
            if r == 0:
                map_values.append(0.0)
                continue
            sum_prec, relevant_count = 0.0, 0
            for idx, rel_val in enumerate(rel):
                if rel_val == 2:
                    relevant_count += 1
                    sum_prec += relevant_count / (idx + 1)
            map_val = sum_prec / r
            map_values.append(map_val)
        return sum(map_values) / self.num_pairs if self.num_pairs else 0.0

    @cached_property
    def successrate_at_1(self) -> float:
        """
        计算所有序列对的 SuccessRate@1 平均值，并缓存结果。

        Returns:
            float: SuccessRate@1 平均值。
        """
        return self.calculate_successrate_at_k(1)

    def calculate_successrate_at_k(self, k: int) -> float:
        """
        计算 SuccessRate@k 平均值。

        Args:
            k (int): 考察的前 k 个候选项数。

        Returns:
            float: SuccessRate@k 平均值。
        """
        sr_values: List[float] = [
            1.0 if any(r == 2 for r in rel[:k]) else 0.0
            for rel in self.relevance
        ]
        return sum(sr_values) / self.num_pairs if self.num_pairs else 0.0

    def calculate_precision_at_k(self, k: int) -> float:
        """
        计算 Precision@k 平均值。

        Args:
            k: 考察的前 k 个候选项数。
        Returns:
            float: Precision@k 平均值。
        """
        prec_values: List[float] = []
        for rel in self.relevance:
            relevant = sum(1 for r in rel[:k] if r == 2)
            prec_values.append(relevant / k if k > 0 else 0.0)
        return sum(prec_values) / self.num_pairs if self.num_pairs else 0.0

    def calculate_recall_at_k(self, k: int) -> float:
        """
        计算 Recall@k 平均值。

        Args:
            k: 考察的前 k 个候选项数。
        Returns:
            float: Recall@k 平均值。
        """
        recall_values: List[float] = []
        for rel, ans in zip(self.relevance, self.answer_lists):
            total = len(ans)
            if total == 0:
                recall_values.append(0.0)
                continue
            relevant = sum(1 for r in rel[:k] if r == 2)
            recall_values.append(relevant / total)
        return sum(recall_values) / self.num_pairs if self.num_pairs else 0.0

    def calculate_ndcg_at_k(self, k: int) -> float:
        """
        计算 NDCG@k 平均值。

        Args:
            k: 考察的前 k 个候选项数。
        Returns:
            float: NDCG@k 平均值。
        """
        ndcg_values: List[float] = []
        for rel, ans in zip(self.relevance, self.answer_lists):
            total = len(ans)
            if total == 0:
                ndcg_values.append(0.0)
                continue
            # 计算 DCG
            dcg = sum(
                (2**val-1) / math.log2(i + 2)
                for i, val in enumerate(rel[:k])
            )
            # 计算理想 DCG
            ideal = sorted(rel, reverse=True)
            idcg = sum(
                (2**val-1) / math.log2(i + 2)
                for i, val in enumerate(ideal[:k])
            )
            ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)
        return sum(ndcg_values) / self.num_pairs if self.num_pairs else 0.0

    def calculate_metrics_for_multiple_k(
        self,
        k_values: List[int]
    ) -> MetricsResult:
        """
        计算序列在给定Ks下的所有指标值。

        Args:
            k_values (List[int]): 给定Ks

        Returns:
            MetricsResult: 所有指标值
        """
        results = {
            "mrr": self.mrr,
            "bleu": self.bleu,
            "map": self.map,
            "successrate_at_ks": [],
            "precision_at_ks": [],
            "recall_at_ks": [],
            "ndcg_at_ks": []
        }
        for k in k_values:
            results["successrate_at_ks"].append(self.calculate_successrate_at_k(k))
            results["precision_at_ks"].append(self.calculate_precision_at_k(k))
            results["recall_at_ks"].append(self.calculate_recall_at_k(k))
            results["ndcg_at_ks"].append(self.calculate_ndcg_at_k(k))
        return MetricsResult(**results)

    def __len__(self) -> int:
        """
        支持内置 len()，返回序列对的数量。

        Returns:
            int: self.num_pairs
        """
        return self.num_pairs

    def __repr__(self) -> str:
        """
        返回 Calculator 实例的简要描述。

        Returns:
            str: 包含 num_pairs 和输入列表长度的信息。
        """
        return (
            f"Calculator(num_pairs={self.num_pairs}, "
            f"seq_lists_length={len(self.seq_lists)}, "
            f"answer_lists_length={len(self.answer_lists)})"
        )
