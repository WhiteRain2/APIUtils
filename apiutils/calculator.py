# SPDX-License-Identifier: MIT

"""
Module: Calculator

提供信息检索和生成模型输出的评估指标计算，包括 MRR、BLEU、MAP、Success@1、
Precision@k、Recall@k、NDCG@k 以及批量多 k 值一次性计算。
"""

import math
from typing import List, Tuple, Dict, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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
        seq_lists: List[List[str]],
        answer_lists: List[List[str]]
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

        self.seq_lists: List[List[str]] = seq_lists
        self.answer_lists: List[List[str]] = answer_lists
        self.num_pairs: int = len(seq_lists)
        # 计算相关性矩阵
        self.relevance: List[List[int]] = self.compute_relevance()
        # 缓存属性
        self._map_avg: Optional[float] = None  # MAP 平均值缓存
        self._success_at_1_avg: Optional[float] = None  # Success@1 平均值缓存

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
                # 前缀匹配得分 1
                elif any(api[:api.rfind('.')] in a for a in ans):
                    rel.append(1)
                else:
                    rel.append(0)
            relevance.append(rel)
        return relevance

    @property
    def mrr(self) -> float:
        """
        计算所有序列对的 MRR（Mean Reciprocal Rank）平均值。

        Returns:
            float: MRR@1 平均值。
        """
        mrr_values: List[float] = []
        # 对每组相关性取第一个完全匹配的倒数
        for rel in self.relevance:
            ranks = [i + 1 for i, r in enumerate(rel) if r == 2]
            mrr = 1.0 / ranks[0] if ranks else 0.0
            mrr_values.append(mrr)
        return sum(mrr_values) / self.num_pairs if self.num_pairs else 0.0

    @property
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
                bleu_val = sentence_bleu([ans], seq, smoothing_function=smoothie)
                bleu_values.append(bleu_val)
        return sum(bleu_values) / self.num_pairs if self.num_pairs else 0.0

    @property
    def map(self) -> float:
        """
        计算所有序列对的 MAP（Mean Average Precision）平均值，并缓存结果。

        Returns:
            float: MAP 平均值。
        """
        if self._map_avg is not None:
            return self._map_avg

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
        self._map_avg = sum(map_values) / self.num_pairs if self.num_pairs else 0.0
        return self._map_avg

    @property
    def success_at_1(self) -> float:
        """
        计算所有序列对的 Success@1 平均值，并缓存结果。

        Returns:
            float: Success@1 平均值。
        """
        if self._success_at_1_avg is not None:
            return self._success_at_1_avg
        success_values: List[float] = [
            1.0 if rel and rel[0] == 2 else 0.0
            for rel in self.relevance
        ]
        self._success_at_1_avg = sum(success_values) / self.num_pairs if self.num_pairs else 0.0
        return self._success_at_1_avg

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
                1.0 / math.log2(i + 2)
                for i, val in enumerate(rel[:k]) if val == 2
            )
            # 计算理想 DCG
            ideal = sorted(rel, reverse=True)
            idcg = sum(
                1.0 / math.log2(i + 2)
                for i, val in enumerate(ideal[:k]) if val == 2
            )
            ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)
        return sum(ndcg_values) / self.num_pairs if self.num_pairs else 0.0

    def calculate_metrics_for_multiple_k(
        self,
        k_values: List[int]
    ) -> Dict[str, List[float]]:
        """
        批量计算多种 k 值下的所有指标。

        Args:
            k_values: 要计算的 k 值列表。
        Returns:
            Dict[str, List[float]]: 各指标在每个 k 下的平均值。
        """
        results: Dict[str, List[float]] = {
            "MRR": [self.mrr],
            "BLEU": [self.bleu],
            "MAP": [self.map],
            "SuccessRate@ks": [],
            "Precision@ks": [],
            "Recall@ks": [],
            "NDCG@ks": []
        }
        for k in k_values:
            # Success@k
            success = [1.0 if any(r == 2 for r in rel[:k]) else 0.0 for rel in self.relevance]
            results["SuccessRate@ks"].append(sum(success) / self.num_pairs)
            # Precision@k
            prec = [sum(1 for r in rel[:k] if r == 2) / k if k > 0 else 0.0 for rel in self.relevance]
            results["Precision@ks"].append(sum(prec) / self.num_pairs)
            # Recall@k
            rec = [
                (sum(1 for r in rel[:k] if r == 2) / len(ans)) if len(ans) > 0 else 0.0
                for rel, ans in zip(self.relevance, self.answer_lists)
            ]
            results["Recall@ks"].append(sum(rec) / self.num_pairs)

            # NDCG@k
            def ndcg_for(rel: List[int]) -> float:
                dcg = sum(1.0 / math.log2(i + 2) for i, v in enumerate(rel[:k]) if v == 2)
                ideal = sorted(rel, reverse=True)
                idcg = sum(1.0 / math.log2(i + 2) for i, v in enumerate(ideal[:k]) if v == 2)
                return dcg / idcg if idcg > 0 else 0.0
            ndcgs = [ndcg_for(rel) for rel in self.relevance]
            results["NDCG@ks"].append(sum(ndcgs) / self.num_pairs)
        return results

    def __len__(self) -> int:
        """
        支持内置 len()，返回序列对的数量。

        Returns:
            int: self.num_pairs
        """
        return self.num_pairs

    def __iter__(self):
        """
        支持迭代协议，使实例可用于 for … in 语句，
        每次迭代返回一个 (seq, ans) 元组。
        """
        for seq, ans in zip(self.seq_lists, self.answer_lists):
            yield seq, ans

    def __contains__(self, item: Tuple[List[str], List[str]]) -> bool:
        """
        支持 in 操作：检查 (seq_list, answer_list) 是否在实例中。

        Args:
            item: 形如 (seq_list, answer_list) 的元组。
        Returns:
            bool: 若存在完全相同的对则返回 True，否则 False。
        """
        # 利用 zip 临时组合，再判断成员
        return item in zip(self.seq_lists, self.answer_lists)

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
