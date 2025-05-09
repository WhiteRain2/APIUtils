import pytest
import math
from apiutils.calculator import Calculator, MetricsResult


class TestCalculator:
    # 基础测试数据
    @pytest.fixture
    def basic_data(self):
        seq_lists = [
            ["java.util.List", "java.util.ArrayList", "java.util.LinkedList"],
            ["org.example.Test", "java.lang.String", "java.util.Map"],
            [],  # 空序列测试
        ]
        answer_lists = [
            ["java.util.ArrayList", "java.util.List"],
            ["java.lang.String"],
            [],  # 空答案测试
        ]
        return seq_lists, answer_lists

    # 初始化测试
    def test_init(self, basic_data):
        seq_lists, answer_lists = basic_data
        calc = Calculator(seq_lists, answer_lists)
        assert calc.num_pairs == 3
        assert len(calc.relevance) == 3
        assert len(calc) == 3

    # 测试初始化参数验证
    def test_init_validation(self):
        # 测试长度不一致
        with pytest.raises(ValueError, match="must have the same length"):
            Calculator(
                [["java.util.List"], ["java.util.Map"]],
                [["java.util.List"]]
            )

        # 测试序列中有重复元素
        with pytest.raises(ValueError, match="duplicate elements"):
            Calculator(
                [["java.util.List", "java.util.List"]],
                [["java.util.List"]]
            )

        # 测试答案中有重复元素
        with pytest.raises(ValueError, match="duplicate elements"):
            Calculator(
                [["java.util.List"]],
                [["java.util.List", "java.util.List"]]
            )

    # 测试相关性计算
    def test_compute_relevance(self, basic_data):
        seq_lists, answer_lists = basic_data
        calc = Calculator(seq_lists, answer_lists)

        # 检查第一组的相关性评分
        assert calc.relevance[0] == [2, 2, 1]

        # 检查第二组的相关性评分
        assert calc.relevance[1] == [0, 2, 0]  # Test不匹配, String完全匹配, Map不匹配

        # 检查空序列的相关性评分
        assert calc.relevance[2] == []  # 空列表

    # 测试MRR计算
    def test_mrr(self):
        seq_lists = [
            ["java.util.Map", "java.util.List", "java.lang.String"],  # 第2个位置匹配
            ["java.util.ArrayList", "java.lang.String"],  # 第1个位置匹配
            ["java.util.HashMap", "java.util.TreeMap"],  # 无匹配
        ]
        answer_lists = [
            ["java.util.List"],
            ["java.util.ArrayList"],
            ["java.lang.String"],
        ]
        calc = Calculator(seq_lists, answer_lists)

        # 手动计算MRR: (1/2 + 1/1 + 0) / 3 = 0.5
        assert math.isclose(calc.mrr, 0.5, rel_tol=1e-9)

    # 测试BLEU计算
    def test_bleu(self):
        seq_lists = [
            ["hello", "world"],
            ["python", "is", "great"],
            [],  # 空序列
        ]
        answer_lists = [
            ["hello", "world", "!"],
            ["java", "is", "great"],
            ["test"],  # 空序列对应的答案
        ]
        calc = Calculator(seq_lists, answer_lists)

        # BLEU值应在0-1之间
        assert 0 <= calc.bleu <= 1

        # 测试完全匹配的BLEU
        perfect_calc = Calculator([["test"]], [["test"]])
        assert math.isclose(perfect_calc.bleu, 1.0, rel_tol=1e-9)

        # 测试完全不匹配的BLEU
        no_match_calc = Calculator([["test"]], [["different"]])
        assert math.isclose(no_match_calc.bleu, 0.0, rel_tol=1e-9)

    # 测试MAP计算
    def test_map(self):
        seq_lists = [
            ["java.util.List",  "java.lang.String", "java.util.ArrayList"],
            ["java.lang.String",  "java.util.List", "java.util.Map"],
        ]
        answer_lists = [
            ["java.util.List", "java.lang.String"],
            ["java.util.List"],
        ]
        calc = Calculator(seq_lists, answer_lists)

        assert math.isclose(calc.map, 0.75, rel_tol=1e-9)

    # 测试Success@1
    def test_successrate_at_1(self):
        seq_lists = [
            ["java.util.List", "java.util.Map"],  # 首位匹配
            ["java.util.Map", "java.util.List"],  # 首位不匹配
        ]
        answer_lists = [
            ["java.util.List"],
            ["java.util.List"],
        ]
        calc = Calculator(seq_lists, answer_lists)

        # 手动计算: (1 + 0) / 2 = 0.5
        assert math.isclose(calc.successrate_at_1, 0.5, rel_tol=1e-9)

    # 测试Success@k
    def test_calculate_successrate_at_k(self):
        seq_lists = [
            ["java.util.Map", "java.util.List"],  # 第2个位置匹配
            ["java.util.Map", "java.util.Set", "java.util.List"],  # 第3个位置匹配
        ]
        answer_lists = [
            ["java.util.List"],
            ["java.util.List"],
        ]
        calc = Calculator(seq_lists, answer_lists)

        # SuccessRate@1: (0 + 0) / 2 = 0
        assert math.isclose(calc.calculate_successrate_at_k(1), 0.0, rel_tol=1e-9)

        # SuccessRate@2: (1 + 0) / 2 = 0.5
        assert math.isclose(calc.calculate_successrate_at_k(2), 0.5, rel_tol=1e-9)

        # SuccessRate@3: (1 + 1) / 2 = 1.0
        assert math.isclose(calc.calculate_successrate_at_k(3), 1.0, rel_tol=1e-9)

    # 测试Precision@k
    def test_calculate_precision_at_k(self):
        seq_lists = [
            ["java.util.List", "java.util.Map", "java.lang.String"],  # 2个匹配
            ["java.util.Set", "java.util.Map", "java.nio.Buffer"],  # 0个匹配
        ]
        answer_lists = [
            ["java.util.List", "java.lang.String"],
            ["java.lang.Object"],
        ]
        calc = Calculator(seq_lists, answer_lists)

        # Precision@2: (1/2 + 0/2) / 2 = 0.25
        assert math.isclose(calc.calculate_precision_at_k(2), 0.25, rel_tol=1e-9)

        # Precision@3: (2/3 + 0/3) / 2 = 1/3
        assert math.isclose(calc.calculate_precision_at_k(3), 1/3, rel_tol=1e-9)

        # 测试k=0的边界情况
        assert math.isclose(calc.calculate_precision_at_k(0), 0.0, rel_tol=1e-9)

    # 测试Recall@k
    def test_calculate_recall_at_k(self):
        seq_lists = [
            ["java.util.List", "java.util.Map", "java.lang.String"],  # 2个匹配，总共2个答案
            ["java.util.Set", "java.lang.Object", "java.nio.Buffer"],  # 1个匹配，总共1个答案
        ]
        answer_lists = [
            ["java.util.List", "java.lang.String"],
            ["java.lang.Object"],
        ]
        calc = Calculator(seq_lists, answer_lists)

        # Recall@1: (1/2 + 0/1) / 2 = 0.25
        assert math.isclose(calc.calculate_recall_at_k(1), 0.25, rel_tol=1e-9)

        # Recall@2: (1/2 + 1/1) / 2 = 0.75
        assert math.isclose(calc.calculate_recall_at_k(2), 0.75, rel_tol=1e-9)

        # Recall@3: (2/2 + 1/1) / 2 = 1.0
        assert math.isclose(calc.calculate_recall_at_k(3), 1.0, rel_tol=1e-9)

        # 测试空答案的情况
        empty_calc = Calculator([["java.util.List"]], [[]])
        assert math.isclose(empty_calc.calculate_recall_at_k(1), 0.0, rel_tol=1e-9)

    # 测试NDCG@k
    def test_calculate_ndcg_at_k(self):
        # 创建一个简单的测试用例，第一个位置完全匹配，其余不匹配
        seq_lists = [
            ["java.util.ArrayList", "java.util.Set", "java.util.Map"],
        ]
        answer_lists = [
            ["java.util.ArrayList"],
        ]
        calc = Calculator(seq_lists, answer_lists)

        # 手动计算NDCG@1:
        # DCG = (2^2-1)/log2(1+1) = 3/1 = 3
        # IDCG = 3
        # NDCG = 3/3 = 1.0
        assert math.isclose(calc.calculate_ndcg_at_k(1), 1.0, rel_tol=1e-9)

        # 测试带前缀匹配的NDCG计算
        seq_lists = [
            ["java.util.List", "java.util.ArrayList", "java.lang.String"],
        ]
        answer_lists = [
            ["java.util.ArrayList"],
        ]
        calc = Calculator(seq_lists, answer_lists)

        # 期望第一个得分为1（前缀匹配），第二个为2（完全匹配）
        # DCG@2 = (2^1-1)/log2(1+1) + (2^2-1)/log2(2+1) = 1 + 3/log2(3)
        # IDCG@2 = (2^2-1)/log2(1+1) + (2^1-1)/log2(2+1) = 3 + 1/log2(3)
        # NDCG@2 = DCG/IDCG
        ndcg2 = calc.calculate_ndcg_at_k(2)
        assert 0 <= ndcg2 <= 1.0

    # 测试多个k值的指标计算
    def test_calculate_metrics_for_multiple_k(self, basic_data):
        seq_lists, answer_lists = basic_data
        calc = Calculator(seq_lists, answer_lists)

        # 计算多个k值的指标
        metrics = calc.calculate_metrics_for_multiple_k([1, 2, 3])

        # 检查结果类型
        assert isinstance(metrics, MetricsResult)

        # 检查结果包含所有必要字段
        assert hasattr(metrics, 'mrr')
        assert hasattr(metrics, 'bleu')
        assert hasattr(metrics, 'map')
        assert hasattr(metrics, 'successrate_at_ks')
        assert hasattr(metrics, 'precision_at_ks')
        assert hasattr(metrics, 'recall_at_ks')
        assert hasattr(metrics, 'ndcg_at_ks')

        # 检查列表长度
        assert len(metrics.successrate_at_ks) == 3
        assert len(metrics.precision_at_ks) == 3
        assert len(metrics.recall_at_ks) == 3
        assert len(metrics.ndcg_at_ks) == 3

    # 测试边界情况
    def test_edge_cases(self):
        # 测试空输入
        empty_calc = Calculator([], [])
        assert empty_calc.num_pairs == 0
        assert empty_calc.mrr == 0.0
        assert empty_calc.bleu == 0.0
        assert empty_calc.map == 0.0

        # 测试计算空k列表的情况
        metrics = empty_calc.calculate_metrics_for_multiple_k([])
        assert len(metrics.successrate_at_ks) == 0
        assert len(metrics.precision_at_ks) == 0
        assert len(metrics.recall_at_ks) == 0
        assert len(metrics.ndcg_at_ks) == 0

    # 测试dunder方法
    def test_dunder_methods(self, basic_data):
        seq_lists, answer_lists = basic_data
        calc = Calculator(seq_lists, answer_lists)

        # 测试__len__
        assert len(calc) == 3

        # 测试__repr__
        repr_string = repr(calc)
        assert "Calculator" in repr_string
        assert "num_pairs=3" in repr_string
        assert "seq_lists_length=3" in repr_string
        assert "answer_lists_length=3" in repr_string
