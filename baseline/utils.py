import sklearn
import numpy as np
from sklearn.metrics import accuracy_score
import pickle as pickle
import pandas as pd
import torch
import re
   

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return (
        sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices)
        * 100.0
    )


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take(torch.tensor([c], dtype=torch.long)).ravel()
        preds_c = probs.take(torch.tensor([c], dtype=torch.long)).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c
        )
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


def label_to_num(label):
    num_label = []
    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


# no entity marker
# def preprocessing_dataset(dataset):
#     """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
#     subject_entity = []
#     object_entity = []
#     for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
#         i = i[1:-1].split(",")[0].split(":")[1]
#         j = j[1:-1].split(",")[0].split(":")[1]


#         subject_entity.append(i)
#         object_entity.append(j)
#     out_dataset = pd.DataFrame(
#         {
#             "id": dataset["id"],
#             "sentence": dataset["sentence"],
#             "subject_entity": subject_entity,
#             "object_entity": object_entity,
#             "label": dataset["label"],
#         }
#     )
#     return out_dataset
def preprocessing_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    entity_sentence = []
    subject_entity_span = []
    object_entity_span = []
    for sen, i, j in zip(
        dataset["sentence"], dataset["subject_entity"], dataset["object_entity"]
    ):
        subj_start_idx = int(
            re.sub(r"[^0-9]", "", i[1:-1].split("'start_idx': ")[1][0:3])
        )
        subj_end_idx = int(re.sub(r"[^0-9]", "", i[1:-1].split("'end_idx': ")[1][0:3]))
        subj_type = i[-5:-2]
        # subj.append((subj_start_idx,subj_end_idx,subj_type))
        
        subj_extra_len = 3 + len(subj_type)
        subject_entity_span.append((subj_start_idx+subj_extra_len,subj_end_idx+subj_extra_len+1))
        obj_start_idx = int(
            re.sub(r"[^0-9]", "", j[1:-1].split("'start_idx': ")[1][0:3])
        )
        obj_end_idx = int(re.sub(r"[^0-9]", "", j[1:-1].split("'end_idx': ")[1][0:3]))
        obj_type = j[-5:-2]
        
        obj_extra_len = subj_extra_len + 4 + len(obj_type)
        object_entity_span.append((obj_start_idx+obj_extra_len, obj_end_idx+obj_extra_len+1))
        i = i[1:-1].split(",")[0].split(":")[1]
        j = j[1:-1].split(",")[0].split(":")[1]

        if subj_start_idx < obj_start_idx:
            sen = (
                sen[:subj_start_idx]
                + "@"
                + "*"
                + subj_type
                + "*"
                + sen[subj_start_idx : subj_end_idx + 1]
                + "@"
                + sen[subj_end_idx + 1 :]
            )
            sen = (
                sen[: obj_start_idx + 7]
                + "#"
                + "^"
                + obj_type
                + "^"
                + sen[obj_start_idx + 7 : obj_end_idx + 7 + 1]
                + "#"
                + sen[obj_end_idx + 7 + 1 :]
            )
        else:
            sen = (
                sen[:obj_start_idx]
                + "#"
                + "^"
                + obj_type
                + "^"
                + sen[obj_start_idx : obj_end_idx + 1]
                + "#"
                + sen[obj_end_idx + 1 :]
            )
            sen = (
                sen[: subj_start_idx + 7]
                + "@"
                + "*"
                + subj_type
                + "*"
                + sen[subj_start_idx + 7 : subj_end_idx + 7 + 1]
                + "@"
                + sen[subj_end_idx + 7 + 1 :]
            )

        entity_sentence.append(sen)
        subject_entity.append(i)
        object_entity.append(j)

    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": entity_sentence,
            "subject_entity": subject_entity,
            "subject_entity_span": subject_entity_span,
            "object_entity": object_entity,
            "object_entity_span": object_entity_span,
            "label": dataset["label"],
        }
    )
    return out_dataset


def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def tokenized_dataset(dataset, tokenizer):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    return tokenized_sentences


def emb_tokenized_dataset(dataset, tokenizer):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )

    entity_loc_ids = []
    for input_token, e01, e02 in zip(
        tokenized_sentences["input_ids"],
        dataset["subject_entity"],
        dataset["object_entity"],
    ):
        subj_token_ids = torch.Tensor(tokenizer(e01)["input_ids"][2:-2])
        obj_token_ids = torch.Tensor(tokenizer(e02)["input_ids"][2:-2])

        subj_start_ids = []
        for idx in range(len(input_token) - len(subj_token_ids)):
            if torch.equal(
                input_token[idx : idx + len(subj_token_ids)], subj_token_ids
            ):
                subj_start_ids.append((idx, len(subj_token_ids)))
                if len(subj_start_ids) == 2:
                    break

        obj_start_ids = []
        for idx in range(len(input_token) - len(obj_token_ids)):
            if torch.equal(input_token[idx : idx + len(obj_token_ids)], obj_token_ids):
                obj_start_ids.append((idx, len(obj_token_ids)))
                if len(obj_start_ids) == 2:
                    break

        entity_loc = [0] * len(input_token)
        for subj_start in subj_start_ids:
            start, length = subj_start
            for idx in range(start, start + length):
                entity_loc[idx] = 1

        for obj_start in obj_start_ids:
            start, length = obj_start
            for idx in range(start, start + length):
                entity_loc[idx] = 2

        entity_loc_ids.append(entity_loc)

    tokenized_sentences["entity_loc_ids"] = torch.tensor(
        entity_loc_ids, dtype=torch.int32
    )

    return tokenized_sentences

def luke_tokenized_dataset(dataset, tokenizer):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    entity_spans = []
    
    for subj_span, obj_span in zip(dataset["subject_entity_span"], dataset["object_entity_span"]):
        entity_spans.append([subj_span, obj_span])
        
    tokenized_sentences = tokenizer(
        list(dataset["sentence"]),
        entity_spans=entity_spans,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    return tokenized_sentences

# conf 를 validation 하는 함수
def validate_conf(conf):
    # Check other parameters as needed
    assert conf.params.batch_size > 0, "batch_size must be greater than 0"
    assert conf.params.max_epoch > 0, "max_epoch must be greater than 0"
    assert isinstance(conf.params.shuffle, bool), "shuffle must be a boolean"
    assert isinstance(conf.params.learning_rate, float), "learning_rate must be a float"
    assert conf.params.learning_rate > 0, "learning_rate must be greater than 0"
    assert isinstance(conf.params.weight_decay, float), "weight_decay must be a float"
    assert conf.params.weight_decay >= 0, "weight_decay must be greater than or equal to 0"
    assert isinstance(conf.params.project_name, str), "project_name must be a string"
    assert isinstance(conf.params.test_name, str), "test_name must be a string"
    assert conf.params.num_labels > 0, "num_labels must be greater than 0"
    assert conf.params.warmup_steps > 0, "warmup_steps must be greater than 0"
    assert isinstance(conf.params.warmup_ratio, float), "warmup_ratio must be a float"
    assert 0 <= conf.params.warmup_ratio <= 1, "warmup_ratio must be between 0 and 1"
    assert conf.params.loss_type in ["cross_entropy", "focal", "label_smoothing"], "loss_type must be one of 'cross_entropy', 'focal', or 'label_smoothing'"
    assert conf.params.classifier in ["default", "LSTM"], "classifier must be either 'default' or 'LSTM'"
    assert isinstance(conf.params.emb, bool), "emb must be a boolean"
    assert conf.params.lr_decay in ["default", "exp"], "lr_decay must be either 'default' or 'exp'"
    assert isinstance(conf.params.use_stratified_kfold, bool), "use_stratified_kfold must be a boolean"
    assert isinstance(conf.params.num_folds, int) and conf.params.num_folds > 0, "num_folds must be a positive integer"
    assert isinstance(conf.params.seed, int), "seed must be an integer"

    # Print success message with separators
    print("*" * 50)
    print("Configuration validation successful.")
    print("*" * 50)