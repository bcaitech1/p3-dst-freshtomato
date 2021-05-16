import json
import argparse
from eval_utils import DSTEvaluator
from collections import defaultdict


SLOT_META_PATH = "data/train_dataset/slot_meta.json"

def _evaluation(preds, labels, slot_meta):
    evaluator = DSTEvaluator(slot_meta)

    evaluator.init()
    assert len(preds) == len(labels)

    for k, l in labels.items():
        p = preds.get(k)
        if p is None:
            raise Exception(f"{k} is not in the predictions!")
        evaluator.update(l, p)

    result = evaluator.compute()
    print(result)

    # domain_slot 별로 정답을 얼마나 맞췄는지 metric 측정을 위한 check dictionary
    acc_dict_per_domain_slot = defaultdict(list)
    for dialog_turn_key, pred_values in preds.items():
        dialog_turn_labels = labels[dialog_turn_key]
        # 실제 label에는 값이 존재하지만, prediction 결과에 none으로 예측한 결과가 생략되어 있어서, (pred에 있지만 label에 없는, 반대의 경우도 있음)
        # 이런 slot들 부분에 대하여서도 정확도를 측정하기 위해 prediction 결과에 "domain-slot-none"으로 추가를 해줌.
        lab_slot_names = set("-".join(lab.split('-')[:2]) for lab in dialog_turn_labels)
        pred_slot_names = set("-".join(pred.split('-')[:2]) for pred in pred_values)
        if lab_slot_names!=pred_slot_names:
            dont_have_in_pred = lab_slot_names-pred_slot_names
            pred_values.extend([f"{extra_slot_pred}-none" for extra_slot_pred in dont_have_in_pred])

            dont_have_in_lab = pred_slot_names-lab_slot_names
            dialog_turn_labels.extend([f"{extra_slot_lab}-none" for extra_slot_lab in dont_have_in_lab])

        # 비교를 위해 dict type으로 바꿔주기
        pred_values_dict = {"-".join(P.split('-')[:2]):P.split('-')[-1] for P in pred_values}
        labels_dict = {"-".join(L.split('-')[:2]):L.split('-')[-1] for L in dialog_turn_labels}
        assert (pred_values_dict.keys()==labels_dict.keys())

        for domain_slot_name, pred in pred_values_dict.items():
            # print(pred_values_dict.keys())
            # print(labels_dict.keys())
            lab = labels_dict[domain_slot_name]
            acc_dict_per_domain_slot[domain_slot_name].append(pred==lab)
    
    acc_dict_per_domain_slot = {f"{k}_accuracy":sum(v)/len(v) for k,v in acc_dict_per_domain_slot.items()}
    result.update(acc_dict_per_domain_slot)
    return result


def evaluation(gt_path, pred_path):
    slot_meta = json.load(open(SLOT_META_PATH))
    gts = json.load(open(gt_path))
    preds = json.load(open(pred_path))
    eval_result = _evaluation(preds, gts, slot_meta)
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    args = parser.parse_args()
    eval_result = evaluation(args.gt_path, args.pred_path)
