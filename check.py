import pandas as pd

from GNNMethod.encoder.prepare_data import Vocabulary

csv_path = "replication.csv"
df = pd.read_csv(csv_path)

type0 = df[df['type'] == 0]
type1 = df[df['type'] == 1]
type2 = df[df['type'] == 2]
type3 = df[df['type'] == 3]
type4 = df[df['type'] == 4]

print("Type 0: ", len(type0))
print("Type 1: ", len(type1))
print("Type 2: ", len(type2))
print("Type 3: ", len(type3))
print("Type 4: ", len(type4))

positive = len(type1) + len(type2) + len(type3) + len(type4)

negative = len(type0)

total = positive + negative

print("total: ", total)

print("positive: ", positive)
print("negative: ", negative)

txt_path = "corpus_deduplicated.txt"
tokenizer = "tokenizer.pkl"

with open(txt_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("lines: ", len(lines))
