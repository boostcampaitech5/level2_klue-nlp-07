from dataloader import Dataloader
from model import Model
import pytorch_lightning as pl
from utils import num_to_label
import pandas as pd
import torch.nn.functional as F
from omegaconf import OmegaConf
import numpy as np
import pickle
from scipy.special import softmax


# Prediction 결과를 기반으로 예측 라벨과 예측 확률 생성
def get_labels_and_probs(prediction):
    preds = []
    probs = []
    for item in prediction:
        pred = item["preds"].tolist()
        prob = item["probs"]
        preds += pred
        probs += prob
    preds = num_to_label(preds)
    probs = [F.softmax(prob, dim=-1).tolist() for prob in probs]
    return preds, probs


# trainer로 예측 결과 구하는 함수
def get_prediction(model, only_entity):
    dataloader = Dataloader(
        model_name=conf.model_name,
        batch_size=conf.params.batch_size,
        shuffle=conf.params.shuffle,
        origin_train_path=None,  # Predict 시엔 None
        train_path=conf.path.train_path,
        dev_path=conf.path.dev_path,
        test_path=conf.path.test_path,
        predict_path=conf.path.predict_path,
        emb=conf.params.emb,
        only_entity=only_entity,
        use_stratified_kfold=None,  # Predict 시엔 None
        train_indices=None,  # Predict 시엔 None
        val_indices=None,  # Predict 시엔 None
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=conf.params.max_epoch,
        log_every_n_steps=1,
        logger=False,
    )

    prediction = trainer.predict(model=model, datamodule=dataloader)

    return prediction


if __name__ == "__main__":
    conf = OmegaConf.load("./config.yaml")

    model = Model.load_from_checkpoint(
        "./ckpt/roberta-large-emb-lr_sched(exp)-kfold4-epoch=04-val_micro_f1=86.36.ckpt"
    )

    klue_prediction = get_prediction(model=model, only_entity=False)
    klue_preds, klue_probs = get_labels_and_probs(klue_prediction)

    entity_prediction = get_prediction(model=model, only_entity=True)
    entity_preds, mask_1 = get_labels_and_probs(entity_prediction)
    mask_2 = []
    for _ in range(len(mask_1)):
        mask_2.append([1] + [0] * 29)

    mask_1 = np.array(mask_1)
    mask_2 = np.array(mask_2)
    with open("./test_label_constraints.pkl", "rb") as f:
        label_constraints = pickle.load(f)

    # 예시값입니다!
    lamb_1 = -1.6
    lamb_2 = 0.3

    new_probs = klue_probs + lamb_1 * mask_1 + lamb_2 * mask_2 + label_constraints

    new_probs = softmax(new_probs, axis=1)
    new_preds = num_to_label(new_probs.argmax(-1))
    new_probs = new_probs.tolist()

    submission = pd.read_csv("./submission/sample_submission.csv")
    submission["pred_label"] = new_preds
    submission["probs"] = new_probs
    submission.to_csv("./submission/core_result.csv", index=False)
