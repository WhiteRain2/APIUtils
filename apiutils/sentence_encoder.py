# SPDX-License-Identifier: MIT

"""
SentenceEncoder 模块

该模块利用SentenceTransformer库对句子进行嵌入编码，并提供查询匹配功能。
"""

import os
import pathlib
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import logging
logger = logging.getLogger(__name__)


class SentenceEncoder:
    """
    SentenceEncoder 提供对句子进行向量化编码、解码以及批量查询匹配功能。
    若需使用GPU加速，请确保安装了CUDA和相应的PyTorch版本。

    Attributes:
        model (SentenceTransformer): 预训练的句嵌入模型。
        queries_embeddings (Dict[Any, np.ndarray]): 存储已编码的查询嵌入。
        queries_dict (Dict[Any, str]): 存储查询 ID 与文本的映射。
    """

    def __init__(
        self,
        model: Any,
        device: str = 'cpu'
    ) -> None:
        """
        初始化 SentenceEncoder 实例。

        Args:
            model: 预训练模型的路径或名称。
            device: 运行设备名称，例如 'cuda' 或 'cpu'。
        """
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA is not available. Switching to CPU.")
            device = 'cpu'
        self.device = device
        # 加载 SentenceTransformer 模型
        self.model: SentenceTransformer = SentenceTransformer(str(model), device=self.device)
        # 存储编码后的查询嵌入
        self.queries_embeddings: Dict[Any, np.ndarray] = {}
        # 存储原始查询文本
        self.queries_dict: Dict[Any, str] = {}

    def encode(
        self,
        sentences: List[str]
    ) -> np.ndarray:
        """
        对一个列表的句子进行嵌入编码。

        Args:
            sentences: 需要编码的句子列表。
        Returns:
            numpy.ndarray: 句子对应的嵌入向量数组。
        """
        return self.model.encode(sentences)

    def decode(
        self,
        embeddings: np.ndarray
    ) -> List[str]:
        """
        将嵌入向量解码为最接近的原始句子（如果模型支持）。

        Args:
            embeddings: 需要解码的嵌入表示。
        Returns:
            List[str]: 对应的句子文本。  # 注意：部分模型可能不支持decode
        """
        return self.model.decode(embeddings)

    def encode_queries(
        self,
        queries_dict: Dict[Any, str]
    ) -> None:
        """
        对查询库中的所有文本进行编码，并保存结果。

        Args:
            queries_dict: 查询 ID 到文本的映射字典。
        """
        self.queries_dict = queries_dict
        items = list(queries_dict.items())
        ids, texts = zip(*items)
        # 显示进度条进行批量编码
        embs: np.ndarray = self.model.encode(list(texts), show_progress_bar=True)
        self.queries_embeddings = dict(zip(ids, embs))

    def get_query_embedding(
        self,
        query_id: Any
    ) -> Optional[np.ndarray]:
        """
        获取指定查询 ID 的嵌入表示。

        Args:
            query_id: 查询在 queries_dict 中对应的键。
        Returns:
            np.ndarray or None: 对应的嵌入向量，如果不存在则返回 None。
        """
        return self.queries_embeddings.get(query_id)

    def save_embeddings(
        self,
        file_path: str | pathlib.Path
    ) -> None:
        """
        将当前 queries_embeddings 和 queries_dict 保存至磁盘文件。

        Args:
            file_path: 目标文件路径（.pkl）。
        """
        if not self.queries_embeddings or not self.queries_dict:
            logger.warning("No embeddings to save.")  # no embeddings
            return
        # 确保目录存在
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump((self.queries_embeddings, self.queries_dict), f)
        logger.info(f"Embeddings and queries saved to {file_path}")  # success message

    def load_embeddings(
        self,
        file_path: str | pathlib.Path
    ) -> None:
        """
        从磁盘加载之前保存的 embeddings 与 queries_dict。

        Args:
            file_path: 包含嵌入的 .pkl 文件路径。
        """
        if not os.path.exists(file_path):
            logger.error("File does not exist.")  # missing file
            return
        with open(file_path, 'rb') as f:
            self.queries_embeddings, self.queries_dict = pickle.load(f)
        logger.info(f"Embeddings and queries loaded from {file_path}")  # success message

    def find_similar_queries(
        self,
        queries: List[str],
        top_k: int
    ) -> List[List[Tuple[Any, float]]]:
        """
        对给定的文本列表，找到在查询库中最相似的 top_k 条查询。

        Args:
            queries: 用户输入的句子列表。
            top_k: 返回最相似查询的数量。
        Returns:
            嵌套列表，外层对应每个输入句子，内层为 (query_id, similarity) 列表。
        """
        # 编码用户输入
        user_embs: np.ndarray = self.encode(queries)
        # 准备查询嵌入矩阵
        all_embs: np.ndarray = np.array(list(self.queries_embeddings.values()))
        # 计算相似度矩阵
        sim_matrix: np.ndarray = cosine_similarity(user_embs, all_embs)

        results: List[List[Tuple[Any, float]]] = []
        for sims in sim_matrix:
            # zip ID 与相似度
            pairs = list(zip(self.queries_embeddings.keys(), sims))
            # 按相似度降序
            pairs.sort(key=lambda x: x[1], reverse=True)
            # 滤除几乎相同的结果
            filtered = [p for p in pairs if p[1] < 0.999]
            results.append(filtered[:top_k])
        return results

    def __call__(
        self,
        sentences: List[str]
    ) -> np.ndarray:
        """
        使实例可直接调用，行为同 encode。

        Args:
            sentences: 需要编码的句子列表。
        Returns:
            嵌入向量数组。
        """
        return self.encode(sentences)

    def __len__(self) -> int:
        """
        返回当前查询库的大小。

        Returns:
            int: queries_dict 的长度。
        """
        return len(self.queries_dict)

    def __bool__(self) -> bool:
        """
        判断当前是否有已加载的查询嵌入。

        Returns:
            bool: 当 queries_dict 和 queries_embeddings 均非空时返回 True。
        """
        return bool(self.queries_dict) and bool(self.queries_embeddings)

    def __repr__(self):
        return f"SentenceEncoder(model={self.model}, device={self.device})"


# Convenience functions

def save_embeddings(
    model: Any,
    queries_dict: Dict[Any, str],
    file_path: str
) -> None:
    """
    快速接口：使用 SentenceEncoder 对象编码并保存 embeddings。

    Args:
        model: 预训练模型的路径或名称。
        queries_dict: 查询字典。
        file_path: 输出文件路径。
    """
    encoder = SentenceEncoder(model)
    encoder.encode_queries(queries_dict)
    encoder.save_embeddings(file_path)


def load_embeddings(
    model: Any,
    pkl_path: str
) -> SentenceEncoder:
    """
    快速接口：加载已保存的 embeddings 并返回对应的 SentenceEncoder 实例。

    Args:
        model: 预训练模型的路径或名称。
        pkl_path: 包含 embeddings 的 .pkl 文件路径。
    Returns:
        SentenceEncoder: 已加载 embeddings 的编码器实例。
    """
    encoder = SentenceEncoder(model)
    encoder.load_embeddings(pkl_path)
    return encoder
