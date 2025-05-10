import pathlib
from itertools import cycle
from typing import Sequence, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt


def draw_liner(x_values: Sequence,
               label_ys_dict: Dict[str, Sequence[np.number | int | float]],
               x_label: str,
               y_label: str,
               save_path: pathlib.Path | str,
               dpi=300,
               title: Optional[str] = None,
               top_offest: float = 0.01,
               ) -> None:
    """
    绘制折线图快速方法，仅支持保存为文件。

    Args:
        x_values (Sequence): x轴数据
        label_ys_dict (Dict[str, Sequence[np.number  |  int  |  float]]): 每条折线的标签及其对应的数据
        x_label (str): x轴标签
        y_label (str): y轴标签
        save_path (pathlib.Path | str): 保存路径
        dpi (int, optional): 保存文件的dpi. Defaults to 300.
        title (Optional[str], optional): 图标题. Defaults to None.
        top_offest (float, optional): y轴上限偏移量. Defaults to 0.01.
    """
    styles_iter = cycle(['g-D', 'r-*', 'y-s', 'c-^'])
    label_font_dict = {
        'family': 'Times New Roman',  # 字体类型（可选）
        'weight': 'bold',  # 加粗
        'size': 16  # 字体大小
    }
    plt.figure()
    for label, y_values in label_ys_dict.items():
        plt.plot(x_values, y_values, next(styles_iter), label=label, linewidth=2)
    plt.grid(True, alpha=0.5)
    bottom, top = plt.ylim()
    plt.ylim(bottom=bottom, top=top+top_offest)
    plt.legend(loc='upper right', ncol=len(label_ys_dict.keys()), frameon=False, handletextpad=0.6)
    if title:
        plt.title(title, fontdict=label_font_dict)
    plt.xlabel(x_label, fontdict=label_font_dict)
    plt.ylabel(y_label, fontdict=label_font_dict)
    plt.savefig(save_path, dpi=dpi)
    plt.close()
