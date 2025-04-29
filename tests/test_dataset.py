# tests/test_dataset.py
import pytest
import pandas as pd
from apiutils import API
from apiutils.dataset import Dataset, DatasetName


# —— 让 API.from_string 可预测 —— #
class DummyAPI:
    @staticmethod
    def from_string(s: str):
        return f"API({s})"


@pytest.fixture(autouse=True)
def patch_api(monkeypatch):
    monkeypatch.setattr(API, 'from_string', DummyAPI.from_string)


# —— 样例 DataFrame —— #
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'title': ['q1', 'q2', 'q3'],
        'answer': ['a1', 'a2', 'a3'],
    })


@pytest.fixture
def biker_dataset():
    return Dataset(DatasetName.BIKER, 'test', 'filtered')


@pytest.fixture
def dataset(sample_df):
    return Dataset.from_dataframe('test_dataset', sample_df)


# —— 测试 from_dataframe —— #
def test_from_dataframe_resets_index_and_sets_name(sample_df):
    df = sample_df.copy()
    df.index = [10, 11, 12]
    ds = Dataset.from_dataframe('custom', df)
    assert ds.name == 'custom'
    assert list(ds.raw.index) == [0, 1, 2]
    assert ds.raw.index.name == 'idx'


def test_from_dataframe_missing_title(sample_df):
    df = sample_df.drop(columns=['title'])
    with pytest.raises(ValueError, match="The question column should be named 'title'"):
        Dataset.from_dataframe('bad', df)


def test_from_dataframe_missing_answer(sample_df):
    df = sample_df.drop(columns=['answer'])
    with pytest.raises(ValueError, match="The answer column should be named 'answer'"):
        Dataset.from_dataframe('bad', df)


# —— 测试 values / titles / answers —— #
def test_values_applies_api_from_string(dataset):
    df_values = dataset.values
    assert isinstance(df_values, pd.DataFrame)
    assert df_values.shape == (3, 2)
    assert list(df_values['answer']) == ['API(a1)', 'API(a2)', 'API(a3)']


def test_titles_and_answers_properties(dataset):
    titles = dataset.titles
    answers = dataset.answers
    assert isinstance(titles, pd.Series)
    assert list(titles) == ['q1', 'q2', 'q3']
    assert isinstance(answers, pd.Series)
    assert list(answers) == ['API(a1)', 'API(a2)', 'API(a3)']


# —— 测试 len, iter, repr —— #
def test_len_and_iter(dataset):
    assert len(dataset) == 3
    rows = list(dataset)
    assert len(rows) == 3
    idx0, row0 = rows[0]
    assert idx0 == 0
    assert row0['title'] == 'q1'


def test_repr(dataset):
    assert repr(dataset) == "Dataset(test_dataset)"


# —— 测试 __getitem__ —— #
def test_getitem_int(dataset):
    row = dataset[1]
    assert isinstance(row, pd.Series)
    assert row['title'] == 'q2'
    assert row['answer'] == 'API(a2)'


def test_getitem_int_out_of_range(dataset):
    with pytest.raises(IndexError):
        _ = dataset[10]


def test_getitem_slice_returns_dataset(dataset):
    sub = dataset[:2]
    assert isinstance(sub, Dataset)
    assert len(sub) == 2
    assert list(sub.titles) == ['q1', 'q2']
    # 原始 dataset 不变
    assert len(dataset) == 3


def test_getitem_slice_index_reset(biker_dataset):
    sub = biker_dataset[1:3]
    assert list(sub.raw.index) == [0, 1]
    assert isinstance(biker_dataset[:3], Dataset)
    assert biker_dataset[:3].titles.equals(biker_dataset.titles[:3])

def test_getitem_invalid_type_raises(dataset):
    with pytest.raises(TypeError, match="Unsupported index type"):
        _ = dataset['title']  # 只支持 int 或 slice
