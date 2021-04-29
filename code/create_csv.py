import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# 데이터셋 불러오기
filepath = "/opt/ml/input/data/train_dataset/train_dials.json"
dataset = json.load(open(filepath))

# 모든 데이터의 state 추출
state_count = defaultdict(int)
for dialogue in dataset:
    for tern in dialogue["dialogue"]:
        if not tern.get("state"):
            continue

        for data in tern["state"]:
            state_count[data] += 1

data_list = []
for key, value in state_count.items():
    d, s, v = key.split("-")
    data_list.append([d, s, v, value])


# DataFrame으로 변환
column_names = ["domain", "slot", "value", "count"]
df = pd.DataFrame(data_list, columns=column_names)

# # csv로 저장
df.to_csv("/opt/ml/input/data/train_dataset/train_dataset.csv")
