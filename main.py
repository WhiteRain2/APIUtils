"""
目前仅用作测试
"""
import os
import pathlib
import pandas as pd
from apiutils.entity import API

root_dir = pathlib.Path(__file__).parent.resolve()


def process_data(data_path):
    df = pd.read_csv(data_path, index_col='idx')
    df['answer'] = df['answer'].apply(lambda a: API.from_string(a))
    e = 0
    for i, ans in enumerate(df['answer']):
        if any(not api.is_standard for api in ans):
            e += 1
        df.loc[i, 'answer'] = ','.join([str(api) for api in ans])
    return df, e


if __name__ == "__main__":
    data_path = root_dir / 'apiutils' / 'dataset'
    for filename in os.listdir(data_path):
        file_path = data_path / filename
        if not os.path.isfile(file_path):
            print(f'The file "{file_path}" is not a file.')
            continue

        df, e = process_data(file_path)
        df.to_csv(file_path, index=True)
        print(f'Processed {filename}, {e} errors found.')

