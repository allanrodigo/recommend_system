from pandas import read_csv, DataFrame
from pathlib import Path

def load_data() -> DataFrame:
    base_dir = Path(__file__).resolve().parent.parent.parent
    input_path = base_dir / 'data' / 'raw' / 'dataset-sales-minor.csv'

    df = read_csv(input_path)
    for i in df.columns:
        mode = df[i].mode()[0]
        df[i].fillna(mode)

    return df
