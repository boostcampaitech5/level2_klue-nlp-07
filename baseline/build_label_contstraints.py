import pandas as pd
from collections import defaultdict
import pickle
import numpy as np

if __name__ == "__main__":
    train = pd.read_csv("../dataset/train/train.csv")
    test = pd.read_csv("../dataset/test/test_data.csv")
    with open("../dict_label_to_num.pkl", "rb") as f:
        label_to_num = pickle.load(f)

    # entity type 별로 가질 수 있는 관계 구하기
    label_dict = defaultdict(set)
    for idx in range(len(train)):
        sbj = train.iloc[idx]["subject_entity"].split("'type': ")[1][1:4]
        obj = train.iloc[idx]["object_entity"].split("'type': ")[1][1:4]
        label = train.iloc[idx]["label"]
        label_dict[(sbj, obj)].add(label)

    # 위에서는 label로만 구했기 때문에, num으로 바꿔준 후 (30, ) 사이즈의 배열로 변환
    label_constraint_dict = {}
    for type_key in label_dict:
        label_nums = [label_to_num[label_] for label_ in label_dict[type_key]]
        const = [0] * 30
        for num in label_nums:
            const[num] = 10
        label_constraint_dict[type_key] = const

    # test dataset에 맞춰 label constraints 모두 구하기
    label_constraints = []
    for idx in range(len(test)):
        sbj = test.iloc[idx]["subject_entity"].split("'type': ")[1][1:4]
        obj = test.iloc[idx]["object_entity"].split("'type': ")[1][1:4]
        try:
            label_constraints.append(label_constraint_dict[(sbj, obj)])
        except:
            label_constraints.append([10] + [0] * 29)

    label_constraints = np.array(label_constraints)

    label_constraints = np.array(label_constraints)
    with open("test_label_constraints.pkl", "wb") as f:
        pickle.dump(label_constraints, f)
