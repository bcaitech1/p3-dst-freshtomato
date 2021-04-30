import os
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_json, save_json
from data_utils import set_seed


NUM_TURNS_Q1 = 12
NUM_TURNS_Q3 = 18


def stratified_subsample(args):
    """층화추출법을 통해 Dialogue 데이터로부터 표본 추출을 수행하는 함수
    층화추출에는 Dialogue별 turn 수와 domain 수를 모두 고려
    => 모집단(기존 데이터)의 분포를 가능한 한 잃지 않도록!

    주어진 학습 데이터의 num_turns(dial별 turn 수)는 하위 25% 분위수와 상위 25% 분위수가 12, 18
    => 각 dial의 num_turns를 '1이상 12 미만', '12 이상 18 미만', '18 이상'의 3가지로 범주화

    이와 더불어 domain수가 최소 1개에서 최대 3개인 것을 고려해, 최종적으로 다음과 같이 9개 범주로 그룹화

    ['1-1', '1-2', '1-3', '2-1' , '2-2', '2-3', '3-1', '3-2', '3-3'] # num_turns 범주 - domain 개수

    부여한 9개 범주를 고려하여 층화추출을 통해 서브샘플링

    - Example
        >>> python code/sampling.py --split_size 0.4 --save_dir './input' --seed 42
        [Extract Meta Info]: 100%|█████████| 7000/7000 [00:00<00:00, 418312.91it/s]
        'train_subsampled_0.4.json' saved in './input'! Size of subsample: 2800
    """
    set_seed(args.seed)

    data = load_json(args.data_dir)
    meta_dict = dict()

    for dial in tqdm(data, desc="[Extract Meta Info]"):
        dial_id = dial["dialogue_idx"]
        num_turns = len(dial["dialogue"])
        num_domains = len(dial["domains"])
        meta_dict[dial_id] = dict(num_turns=num_turns, num_domains=num_domains)

    meta_table = pd.DataFrame(meta_dict).T.reset_index()

    # num_turns 범주화
    g1_indices = meta_table[meta_table["num_turns"] < NUM_TURNS_Q1].index
    g2_indices = meta_table[
        (meta_table["num_turns"] >= NUM_TURNS_Q1)
        & (meta_table["num_turns"] < NUM_TURNS_Q3)
    ].index
    g3_indices = meta_table[meta_table["num_turns"] >= NUM_TURNS_Q3].index

    meta_table["turn_group"] = 0
    meta_table.loc[g1_indices, "turn_group"] = 1
    meta_table.loc[g2_indices, "turn_group"] = 2
    meta_table.loc[g3_indices, "turn_group"] = 3

    # 최종 9개의 범주 부여
    meta_table["group"] = (
        meta_table["turn_group"].apply(str)
        + "-"
        + meta_table["num_domains"].apply(str)
    )
    subsampled_table, _ = train_test_split(
        meta_table, train_size=args.split_size, stratify=meta_table["group"]
    )
    subsample_indices = subsampled_table["index"].tolist()
    subsampled_data = list(
        filter(lambda x: x["dialogue_idx"] in subsample_indices, data)
    )

    fname = f"train_subsampled_{args.split_size}.json"
    save_json(os.path.join(args.save_dir, fname), subsampled_data)
    print(f"'{fname}' saved in '{args.save_dir}'! Size of subsampled data: {len(subsampled_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="표본추출할 Dialogue 데이터 경로",
        default="./input/data/train_dataset/train_dials.json",
    )
    parser.add_argument("--split_size", help="표본추출 비율 설정", default=0.4)
    parser.add_argument("--save_dir", help="저장할 디렉토리", default="./input")
    parser.add_argument("--seed", help="추출 시 고정할 시드값", default=42)

    args = parser.parse_args()
    stratified_subsample(args)
