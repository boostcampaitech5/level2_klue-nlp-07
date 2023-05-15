from dataloader import Dataloader
import argparse
from model import Model
import pytorch_lightning as pl
import torch
from utils import num_to_label
import pandas as pd
import torch.nn.functional as F


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="klue/roberta-large", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epoch", default=10, type=int)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--train_path", default="../dataset/train/train_split.csv")
    parser.add_argument("--dev_path", default="../dataset/dev/dev.csv")
    parser.add_argument("--test_path", default="../dataset/dev/dev.csv")
    parser.add_argument("--predict_path", default="../dataset/test/test_data.csv")
    parser.add_argument("--project_name", default="refactoring")
    parser.add_argument("--test_name", default="roberta-large-focal-entity")
    parser.add_argument("--num_labels", default=30)
    parser.add_argument("--warmup_steps", default=500)
    parser.add_argument("--loss_type", default="focal")
    parser.add_argument("--classifier", default="default")
    parser.add_argument("--emb", default=True)
    parser.add_argument("--lr_decay", default="default")
    args = parser.parse_args(args=[])

    dataloader = Dataloader(
        model_name=args.model_name,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        predict_path=args.predict_path,
        emb=args.emb,
    )

    model = Model(
        model_name=args.model_name,
        lr=args.learning_rate,
        num_labels=args.num_labels,
        warmup_steps=args.warmup_steps,
        loss_type=args.loss_type,
        classifier=args.classifier,
        max_training_step=args.max_epoch * 100,
        lr_decay=args.lr_decay,
    )
    checkpoint = torch.load("./ckpt/roberta-large-epoch=03-val_micro_f1=86.25.ckpt")
    model.load_state_dict(checkpoint["state_dict"])

    trainer = pl.Trainer(
        accelerator="gpu", max_epochs=args.max_epoch, log_every_n_steps=1, logger=False
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
