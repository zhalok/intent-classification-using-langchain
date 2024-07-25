import pandas as pd


def download_dataset():
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'val_iron': 'data/val_iron-00000-of-00001.parquet', 'val_neg': 'data/val_neg-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/jakeazcona/short-text-labeled-emotion-classification/" + splits["test"])
    print(df.to_csv("classification_test_dataset.csv"))

def get_test_dataset():
    test_dataset = pd.read_csv("classification_test_dataset.csv")
    return test_dataset



def pick_random_test(n):
    test_dataset = pd.read_csv("classification_test_dataset.csv")
    return test_dataset.sample(n=n)

