from dataloader import Dataloader
import argparse
from model import Model
import pytorch_lightning as pl
import torch
from utils import num_to_label
import pandas as pd
import torch.nn.functional as F
from omegaconf import OmegaConf


if __name__ == "__main__":
    conf = OmegaConf.load("./config.yaml")

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
        use_stratified_kfold=None,  # Predict 시엔 None
        train_indices=None,  # Predict 시엔 None
        val_indices=None,  # Predict 시엔 None
    )

    model = Model.load_from_checkpoint(
        "./ckpt/roberta-large-emb-lr_sched-epoch=01-val_micro_f1=86.09.ckpt"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=conf.params.max_epoch,
        log_every_n_steps=1,
        logger=False,
    )

    prediction = trainer.predict(model=model, datamodule=dataloader)

    preds = []
    probs = []
    for item in prediction:
        pred = item["preds"].tolist()
        prob = item["probs"]
        preds += pred
        probs += prob
    preds = num_to_label(preds)
    probs = [F.softmax(prob, dim=-1).tolist() for prob in probs]

    submission = pd.read_csv("./submission/sample_submission.csv")
    submission["pred_label"] = preds
    submission["probs"] = probs
    submission.to_csv("./submission/submission.csv", index=False)
