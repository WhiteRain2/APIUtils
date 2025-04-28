[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 简要概述

`APIUtils` 是一个API推荐领域面向科研与开发的 **Python 通用工具包**，旨在提供高效、易用的API调用工具，支持多种API服务的快速集成与使用。
模块使用细节请参阅模块文档。

---
> **注意**：本项目仍在持续更新中，API接口和功能可能会频繁变动，暂时建议以本地开发模式使用。

## 目录

- [安装](#安装)
- [模块说明](#模块说明)
- [配置与依赖](#配置与依赖)
- [贡献指南](#贡献指南)  
- [许可证](#许可证)

---

## 安装

1. **从 GitHub**  
   ```bash
   pip install git+https://github.com/WhiteRain2/APIUtils.git
   ```
   
   建议使用[uv](https://www.datacamp.com/tutorial/python-uv)工具安装:
   ```bash
   uv add git+https://github.com/WhiteRain2/APIUtils.git
   ```

2. **开发模式**  
   ```bash
   git clone https://github.com/WhiteRain2/APIUtils.git
   cd apiutils
   pip install -e .
   ```  
   即可在本地编辑时实时生效。

---

## 模块说明

### Calculator

### dataset

### API

### LLMService

### SentenceEncoder

[此文档待补充，请参阅模块级文档]

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
