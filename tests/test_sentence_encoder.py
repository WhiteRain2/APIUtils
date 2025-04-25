import pytest
from apiutils import SentenceEncoder
import numpy as np
import os


# 使用预训练的小型模型进行测试
@pytest.fixture(scope="module")
def encoder():
    model_name = 'all-MiniLM-L6-v2'
    return SentenceEncoder(model_name)


def test_encode(encoder):
    sentences = ["Hello world", "Goodbye world"]
    embeddings = encoder.encode(sentences)
    assert embeddings.shape == (2, 384)  # 假定模型的嵌入维度是384


def test_encode_queries(encoder):
    queries_dict = {1: "Hello world", 2: "Goodbye world"}
    encoder.encode_queries(queries_dict)
    assert len(encoder.queries_embeddings) == 2


def test_save_and_load_embeddings(encoder, tmp_path):
    queries_dict = {1: "Hello world", 2: "Goodbye world"}
    encoder.encode_queries(queries_dict)
    file_path = tmp_path / "test_embeddings.pkl"
    encoder.save_embeddings(file_path)

    # 创建新的实例来加载
    new_encoder = SentenceEncoder('all-MiniLM-L6-v2')
    new_encoder.load_embeddings(file_path)
    assert len(new_encoder.queries_dict) == 2


def test_find_similar_queries(encoder):
    queries_dict = {1: "Hello world", 2: "Goodbye world"}
    encoder.encode_queries(queries_dict)
    results = encoder.find_similar_queries(["Hello"], 1)
    assert any(result[0][0] == 1 for result in results)  # 检查是否找到了 "Hello world"


def test_call(encoder):
    sentences = ["Hello world"]
    embeddings = encoder(sentences)  # 使用 __call__ 方法
    assert embeddings.shape == (1, 384)
