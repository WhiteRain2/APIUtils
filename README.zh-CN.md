[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | [中文](README.zh-CN.md)

## 简要概述

`APIUtils` 是一个 API 推荐领域面向科研与开发的 **Python 通用工具包**，旨在提供高效、易用的 API 调用工具，支持多种 API 服务的快速集成与使用。
模块使用细节请参阅模块文档。

---

## 目录

- [简要概述](#简要概述)
- [目录](#目录)
- [安装](#安装)
- [快速开始](#快速开始)
  - [API 实体解析与标准化](#api-实体解析与标准化)
  - [句子编码与语义匹配](#句子编码与语义匹配)
  - [LLM 服务调用](#llm-服务调用)
  - [数据集操作](#数据集操作)
  - [评估指标计算](#评估指标计算)
  - [绘制评估图表](#绘制评估图表)
- [模块说明](#模块说明)
  - [Calculator](#calculator)
  - [dataset](#dataset)
  - [API](#api)
  - [LLMService](#llmservice)
  - [SentenceEncoder](#sentenceencoder)
  - [chart](#chart)
- [配置与依赖](#配置与依赖)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 安装

```bash
pip install apiutils-rec
```

建议使用[uv](https://www.datacamp.com/tutorial/python-uv)工具安装:

```bash
uv add apiutils-rec
```

安装最新版本:

```bash
uv add git+https://github.com/WhiteRain2/APIUtils.git
```

---

## 快速开始

### API 实体解析与标准化

```python
from apiutils import API

# 从字符串解析API
api = API("java.util.List.add(Object)")
print(api.fullname)  # java.util.List.add
print(api.method)    # add
print(api.prefix)    # java.util.List
print(api.args)      # ['Object']

# 批量解析多个API
apis = API.from_string("使用java.util.List.add添加元素，然后用java.util.Collections.sort排序")
for api in apis:
    print(f"找到API: {api}")

# 检查是否为标准API
if api.is_standard:
    print("这是标准API")
else:
    # 获取可能的标准API
    possible_apis = api.get_possible_standard_apis()
    print(f"可能的标准API: {possible_apis}")
```

### 句子编码与语义匹配

```python
from apiutils import SentenceEncoder

# 初始化编码器
encoder = SentenceEncoder('all-MiniLM-L6-v2')

# 编码查询
queries_dict = {
    1: "How to add elements to a list?",
    2: "How to sort a collection?",
    3: "Converting string to integer in Java"
}
encoder.encode_queries(queries_dict)

# 查找语义相似的查询
results = encoder.find_similar_queries(
    ["How can I put items into a list?"],
    top_k=2
)
print(results)  # [[(query_id, similarity_score), ...]]

# 保存和加载嵌入
encoder.save_embeddings("embeddings.pkl")
encoder.load_embeddings("embeddings.pkl")
```

### LLM 服务调用

```python
import asyncio
from apiutils import LLMService

# 设置全局客户端配置
LLMService.set_llm_client_config(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"
)

# 创建服务实例
service = LLMService(
    model="gpt-4o-mini",
    system_prompt="你是一个Java API专家",
    configs={"temperature": 0.7}
)

# 异步对话
async def chat_example():
    async for chunk in service.chat("如何在Java中实现线程安全的集合?"):
        print(chunk, end="")

# 批量处理多个查询
async def batch_example():
    questions = [
        "如何在Java中实现单例模式?",
        "Java Stream API怎么使用?",
        "如何处理NullPointerException?"
    ]
    responses = await service.queries(questions, batch_size=2)
    for res in responses:
        print(f"问题: {res.query}\n回答: {res.answer}\nTokens: {res.tokens}\n")

# 运行异步示例
asyncio.run(chat_example())
asyncio.run(batch_example())
```

### 数据集操作

```python
from apiutils.dataset import Dataset, DatasetName

# 加载预定义数据集
biker_dataset = Dataset(DatasetName.BIKER, 'test', 'filtered', nrows=100)
apibench_dataset = Dataset(DatasetName.APIBENCH_Q, 'train', nrows=50)

# 访问数据
for idx, row in enumerate(biker_dataset):
    print(f"问题 {idx}: {row.title}")
    print(f"API答案: {row.answer}")
    if idx >= 2:
        break

# 从DataFrame创建自定义数据集
import pandas as pd
df = pd.DataFrame({
    'title': ["如何将字符串转为整数?", "如何排序列表?"],
    'answer': ["Integer.parseInt()", "Collections.sort()"]
})
custom_dataset = Dataset.from_dataframe("自定义数据集", df)
```

### 评估指标计算

```python
from apiutils import Calculator

# 准备候选序列和参考答案
candidate_lists = [
    ["java.util.List.add", "java.util.ArrayList.size", "java.util.Collections.sort"],
    ["java.lang.Integer.parseInt", "java.lang.Long.parseLong"]
]
reference_lists = [
    ["java.util.List.add", "java.util.Collections.sort"],
    ["java.lang.Integer.parseInt"]
]

# 计算评估指标
calculator = Calculator(candidate_lists, reference_lists)
print(f"MRR: {calculator.mrr:.4f}")
print(f"BLEU: {calculator.bleu:.4f}")
print(f"MAP: {calculator.map:.4f}")

# 计算多个k值的指标
k_values = [1, 3, 5]
metrics = calculator.calculate_metrics_for_multiple_k(k_values)
print(f"Success@{k_values}: {metrics.successrate_at_ks}")
print(f"Precision@{k_values}: {metrics.precision_at_ks}")
print(f"Recall@{k_values}: {metrics.recall_at_ks}")
print(f"NDCG@{k_values}: {metrics.ndcg_at_ks}")
```

### 绘制评估图表

```python
from apiutils.chart import draw_liner
import numpy as np
import pathlib

# 准备数据
x_values = [1, 3, 5, 10]
y_dict = {
    "模型A": [0.75, 0.82, 0.88, 0.92],
    "模型B": [0.68, 0.78, 0.85, 0.90]
}

# 绘制折线图
draw_liner(
    x_values=x_values,
    label_ys_dict=y_dict,
    x_label="K值",
    y_label="Precision@K",
    save_path="precision_comparison.png",
    title="模型性能对比"
)
```

---

## 模块说明

### Calculator

Calculator 模块提供了全面的信息检索与生成模型评估指标计算功能，支持:

- MRR (Mean Reciprocal Rank): 平均倒数排名，评估第一个正确答案的排名质量
- BLEU: 双语评估替换算法，评估生成序列与参考序列的相似度
- MAP: 平均精确率均值，综合评估检索结果的质量
- Success@k: 前 k 项中至少有一个正确答案的比例
- Precision@k: 前 k 项中正确答案占比
- Recall@k: 正确答案中被成功检索到前 k 项的比例
- NDCG@k: 归一化折损累积增益，考虑排序位置的评估指标

可以对 API 推荐模型输出与标准答案进行全面评估，适用于 API 推荐、代码补全等任务。

### dataset

dataset 模块提供了 API 推荐领域常用数据集的访问接口:

- BIKER: 面向 API 推荐的大型问答数据集，包含训练集和测试集
- APIBENCH-Q: API 基准评测数据集，适合评估模型在 API 查询任务上的表现

支持按索引或切片访问数据，转换为 API 实体对象，以及从自定义 DataFrame 创建数据集。

### API

API 模块提供了对 Java API 字符串的解析和标准化功能，支持:

- 从字符串解析 API 全名、方法名、前缀和参数
- 检查 API 是否为标准 API
- 获取可能的标准 API 匹配列表
- 支持自定义标准 API 库

适用于 API 识别、标准化和匹配任务。

### LLMService

LLMService 模块是对 OpenAI API 的高级封装，提供:

- 异步流式对话（支持多轮对话管理）
- 批量并发查询处理（带限流、超时与重试机制）
- 会话历史管理
- Token 使用统计

面向生产环境优化，提供鲁棒的大语言模型访问能力。

### SentenceEncoder

SentenceEncoder 模块利用 SentenceTransformer 库实现句子语义编码与匹配:

- 文本向量化编码与解码
- 批量查询编码与保存
- 相似度计算与语义匹配
- 支持 CPU 和 GPU(CUDA)加速（需自行安装适配的 CUDA 和 Pytorch）

适用于语义检索、问题匹配、文档相似度分析等任务。

### chart

chart 模块提供简单易用的图表绘制功能:

- 多序列折线图绘制
- 支持自定义标签、字体和样式
- 便捷的文件保存接口

适合快速可视化评估结果和实验数据。

---

## 配置与依赖

- Python ≥ 3.8；
- `openai`、`tqdm`、`sentence-transformers`、`scikit-learn`、`nltk` 等⸺详见 `pyproject.toml` 中的 `dependencies` 列表。
- 建议使用 `uv` 工具获得一致性体验。

---

## 贡献指南

欢迎提交 **Issue** 和 **Pull Request**。

1. Fork 本仓库并新建分支；
2. 按 [PEP8](https://peps.python.org/pep-0008/) 和 [Google Style](https://google.github.io/styleguide/pyguide.html) 撰写代码；
3. 参加测试覆盖与文档补充；
4. 提交 PR 并描述改动背景。

---

## 许可证

本项目采用 **MIT License**，详见 [LICENSE](LICENSE) 文件，欢迎自由使用与贡献。
