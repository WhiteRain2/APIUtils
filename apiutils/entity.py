# SPDX-License-Identifier: MIT

"""
API 模块
该模块提供了对API字符串的解析和标准化功能。它支持从字符串创建API对象，检查API是否为标准API，并获取可能的标准API列表。
允许用户自义定标准APIs
"""

import re
import pathlib
import pandas as pd
from typing import Optional, List, Tuple

_src_dir = pathlib.Path(__file__).parent.resolve()


def _get_doc_api_fullname() -> Tuple[List[str], List[str]]:
    """
    获取doc_apis
    :return: doc_apis
    """
    doc = pd.read_csv(_src_dir / 'dataset' / 'doc' / 'doc_APIs_descriptions.csv')[['API', 'description']].to_dict()
    return list(doc['API'].values()), list(doc['description'].values())


class API:
    _standard_api_strings, _standard_api_description = _get_doc_api_fullname()
    _dot_string_pattern = r'\b([a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)+)(\((.*?)\))?'
    _standard_apis = None

    def __init__(self, api_str: str, description: Optional[str] = None) -> None:
        """
        初始化API对象
        :param api_str: str
        """
        self.fullname: str = ''
        self.method: str = ''
        self.prefix: str = ''
        self.args: Optional[list] = None
        self.parts: List[str] = []
        self.description: Optional[str] = description

        match = re.search(API._dot_string_pattern, api_str)
        if match:
            self.fullname = match.group(1)
            if not match.group(3):
                self.args = None
            else:
                self.args = [arg.strip() for arg in match.group(4).split(',') if arg.strip()] if match.group(4) else []
            self.prefix, self.method = self.fullname.rsplit('.', 1)
            self.parts = self.fullname.split('.')
        else:
            raise ValueError(f"Invalid API string: {api_str}")

    @classmethod
    def from_string(cls, api_str: str) -> List['API']:
        """
        从字符串创建API对象
        :param api_str: str
        :return: List[API]
        """
        apis = []
        for match in re.finditer(cls._dot_string_pattern, api_str):
            api_fullname = match.group(1)
            args = match.group(3) or ''
            apis.append(cls(api_fullname + args))
        return apis

    @classmethod
    def set_standard_apis(cls, standard_apis: List[str], standard_api_descriptions: Optional[List[str]]) -> None:
        """
        设置标准API列表，支持有参或无参API
        :param standard_apis: List[str]
        :param standard_api_descriptions: Optional[List[str]]
        """
        # cls._standard_apis = [cls(api_str) for api_str in standard_apis]
        cls._standard_api_strings = standard_apis
        cls._standard_api_description = standard_api_descriptions or [None] * len(standard_apis)
        cls._standard_apis = [cls(api_str, api_doc)
                              for api_str, api_doc in zip(cls._standard_api_strings, cls._standard_api_description)]

    @classmethod
    def get_standard_apis(cls) -> List['API']:
        """
        获取标准API列表
        :return: List[API]
        """
        if not cls._standard_apis:
            cls._standard_apis = [cls(api_str, api_doc)
                                  for api_str, api_doc in zip(cls._standard_api_strings, cls._standard_api_description)]
        return cls._standard_apis

    @property
    def is_standard(self, with_args: bool = False) -> bool:
        """
        判断API是否为标准API
        :param with_args: bool 是否考虑参数，默认为False
        :return: bool
        """
        if not with_args:
            return any(self.fullname == api.fullname for api in self.get_standard_apis())
        else:
            return any(self.fullname == api.fullname and self.args == api.args for api in self.get_standard_apis())

    def get_possible_standard_apis(self, matched_ps: int = 1, first: bool = False) -> List[Optional['API']]:
        """
        自义定匹配阈值，获取可能的标准API列表
        :param first: bool 是否只返回第一个匹配的API，默认为False
        :param matched_ps: int 需要匹配的部分数，默认为1
        :return: List[Optional[API]]
        """
        if matched_ps < 1:
            raise ValueError('numbers of matched part must be greater than 1')
        if self.is_standard:
            return [self]
        if not (self.fullname and self.method):
            raise ValueError('API is not valid')
        p_apis = []
        for standard_api in API.get_standard_apis():
            if self.parts[::-1][:matched_ps+1] == standard_api.parts[::-1][:matched_ps+1]:  # +1 表示方法名也要匹配
                p_apis.append(standard_api)
                if first:
                    break
        return p_apis

    def __eq__(self, other):
        if isinstance(other, API):
            return self.fullname == other.fullname and self.args == other.args
        return False

    def __hash__(self):
        args_tuple = tuple(self.args) if self.args is not None else ()
        return hash((self.fullname, args_tuple))

    def __repr__(self):
        return f"API({str(self)})"

    def __str__(self):
        r = f"{self.fullname}"
        if self.args is not None:
            r += f"({', '.join(self.args)})"
        return r

    def __len__(self):
        return len(self.parts)
