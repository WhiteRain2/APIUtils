# SPDX-License-Identifier: MIT

"""
Dataset 模块
"""

import pathlib
from typing import Literal, Optional
from enum import Enum, auto
import pandas as pd

from .entity import API

_src_dir = pathlib.Path(__file__).parent.resolve()
_data_dir = _src_dir / 'dataset'


class DatasetName(Enum):
    BIKER = auto()
    APIBENCH_Q = auto()


class Dataset:
    """
    常见API领域数据集

    - BIKER Dataset
        - Train: BIKER训练集, 包含33872条QA对
        - Test:
            - Original: BIKER原论文测试集, 包含413条QA对
            - Filtered: 经过人工筛选的BIKER测试集, 包含259条QA对

    - APIBENCH-Q Dataset
        - Train: APIBENCH-Q原始数据集, 包含6563条QA对
        - Test:  Yujia Chen等人筛选的APIBENCH-Q测试集, 包含500条QA对
    """
    _dataset_path_mapper = {
        DatasetName.BIKER: {
            'train': _data_dir / 'BIKER' / 'BIKER_train.csv',
            'test': {
                'original': _data_dir / 'BIKER' / 'BIKER_test.csv',
                'filtered': _data_dir / 'BIKER' / 'BIKER_test_filtered.csv',
            },
        },
        DatasetName.APIBENCH_Q: {
            'train': _data_dir / 'APIBENCH' / 'Q_train.csv',
            'test': _data_dir / 'APIBENCH' / 'Q_test.csv',
        }
    }

    def __init__(self,
                 dataset: Optional[DatasetName],
                 tpe: Literal['train', 'test'],
                 optional: str = None):
        """
        初始化数据集对象
        :param dataset: 数据集
        :param tpe: Literal['train', 'test']
        :param optional: Literal[None, 'original', 'filtered']
        """
        if dataset is None:  # 用于自定义数据
            self._dataset_path = None
            self._original_df = None
            self._values = None
            self.name = None
            return
        try:
            self._dataset_path = self._dataset_path_mapper[dataset][tpe]
            if optional:
                if isinstance(self._dataset_path, dict):
                    self._dataset_path = self._dataset_path[optional]
                else:
                    raise ValueError(f"Optional parameter is not applicable for {dataset} {tpe}")
        except KeyError:
            raise ValueError(f"Invalid dataset name, type or optional: {dataset}, {tpe}, {optional}")
        self._original_df = pd.read_csv(self._dataset_path, index_col='idx')
        self._values = None
        self.name = dataset.name

    @classmethod
    def from_dataframe(cls, name: str, data: pd.DataFrame) -> 'Dataset':
        """
        从DataFrame创建数据集对象
        :param name: str
        :param data: pd.DataFrame
        :return: Dataset
        """
        if not isinstance(data.index, pd.RangeIndex) or data.index.start != 0 or data.index.step != 1:
            data = data.reset_index(drop=True)

        data.index.name = 'idx'
        if 'title' not in data.columns:
            raise ValueError("The question column should be named 'title'")
        if 'answer' not in data.columns:
            raise ValueError("The answer column should be named 'answer'")
        dataset = cls(None, 'train')
        dataset._original_df = data
        dataset.name = name
        return dataset

    @property
    def raw(self) -> pd.DataFrame:
        """
        获取原始数据集, 建议使用values方法获取处理后的数据
        :return: pd.DataFrame
        """
        return self._original_df

    @property
    def values(self) -> pd.DataFrame:
        """
        获取经API实例化的数据
        :return: pd.DataFrame
        """
        if self._values is None:
            self._values = self.raw.assign(
                answer=self.raw['answer'].apply(lambda x: API.from_string(x))
            )
        return self._values

    @property
    def titles(self) -> 'pd.Series[str]':
        """
        获取数据集中的标题
        :return: pd.Series[str]
        """
        return self.values['title']

    @property
    def answers(self) -> 'pd.Series[API]':
        """
        获取数据集中的答案
        :return: pd.Series[API]
        """
        return self.values['answer']

    def __getitem__(self, key):
        return self.values.loc[key]

    def __iter__(self):
        return self.values.iterrows()

    def __len__(self):
        return len(self.values)

    def __str__(self):
        return f"Dataset({self.name})"
