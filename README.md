[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 简要概述

`APIUtils` 是一个API推荐领域面向科研与开发的 **Python 通用工具包**，集成了以下三大核心功能：  
1. **LLMService**：基于 OpenAI 异步接口的对话与批量查询封装；  
2. **SentenceEncoder**：使用 Sentence‑Transformer 进行句子向量化与相似度检索；  
3. **Calculator**：提供 MRR、MAP、Success@k、Precision@k、Recall@k、NDCG@k、BLEU 等评估指标计算。

---

## 目录

- [安装](#安装)  
- [快速开始](#快速开始)  
- [模块说明](#模块说明)  
  - [LLMService](#llmservice)  
  - [SentenceEncoder](#sentenceencoder)  
  - [Calculator](#calculator)  
- [配置与依赖](#配置与依赖)
- [贡献指南](#贡献指南)  
- [许可证](#许可证)

---

## 安装

1. **从 GitHub**  
   ```bash
   pip install git+https://github.com/WhiteRain2/APIUtils.git@v0.1.0#egg=apiutils
   ```  
   该方式可安装指定标签或分支上的版本。

2. **开发模式**  
   ```bash
   git clone https://github.com/WhiteRain2/APIUtils.git
   cd apiutils
   pip install -e .
   ```  
   即可在本地编辑时实时生效。

---

## 快速开始

```python
from apiutils.llm_service import LLMService
from apiutils.sentence_encoder import SentenceEncoder
from apiutils.calculator import Calculator

# 配置 LLM 客户端
LLMService.set_llm_client_config(api_key="YOUR_KEY", base_url="https://api.openai.com")

# 初始化服务
svc = LLMService(model="gpt-4o-mini")
# 单次对话
async for chunk in svc.chat("Hello world", stream=True):
    print(chunk)

# 句子编码
encoder = SentenceEncoder("all-mpnet-base-v2")
emb = encoder(["This is a test."])

# 评估指标
seqs = [["a.b", "a.c"], ["x.y", "z.w"]]
answers = [["a.b"], ["z.w"]]
calc = Calculator(seqs, answers)
print("MRR:", calc.mrr)
```

---

## 模块说明

### LLMService

- **功能**：异步聊天、流式与非流式返回、多任务并发与重试、历史管理。  
- **主要方法**：  
  - `set_llm_client_config(api_key, base_url)`：全局初始化客户端。  
  - `chat(query, stream=True)`：对话生成；  
  - `queries(list_of_queries)`：批量限流并发查询；  
  - `__len__, __iter__, __getitem__`：支持容器协议获取历史记录。  

### SentenceEncoder

- **功能**：调用 `sentence-transformers` 对句子进行嵌入、解码、查询匹配。  
- **主要方法**：  
  - `encode(sentences)` / `decode(embeddings)`；  
  - `encode_queries(dict_of_id2text)`：批量编码并保存映射；  
  - `find_similar_queries(list_of_texts, top_k)`：基于余弦相似度检索。  

### Calculator

- **功能**：对候选序列与答案列表计算 MRR、BLEU、MAP、Success@1、Precision@k、Recall@k、NDCG@k，以及一次性多 k 值指标统计。  
- **协议方法**：支持 `len(calc)`、`for ... in calc`、`calc[i]`、`item in calc` 等 Python 序列协议。  

---

## 配置与依赖

- Python ≥3.8；  
- `openai`、`tqdm`、`sentence-transformers`、`scikit-learn`、`nltk` 等⸺详见 `pyproject.toml` 中的 `dependencies` 列表。  
- 可选安装 `pipx install uv` 以获得极速依赖锁定与同步体验。

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
