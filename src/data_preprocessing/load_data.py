import pandas as pd

def load_raw_data(path: str):
    return pd.read_csv(path)
