from dataloader import Dataloader
import argparse
from model import Model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="klue/roberta-large", type=str)
    # parser.add_argument("--model_name", default="xlm-roberta-large", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epoch", default=10, type=int)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--train_path", default="../dataset/train/train_split.csv")
    parser.add_argument("--dev_path", default="../dataset/dev/dev.csv")
    parser.add_argument("--test_path", default="../dataset/dev/dev.csv")
    parser.add_argument("--predict_path", default="../dataset/test/test_data.csv")
    parser.add_argument("--project_name", default="refactoring")
    parser.add_argument(
        "--test_name", default="roberta-large-emb-lr_sched-focal-entity"
    )
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
        max_training_step=args.max_epoch * 900,
        loss_type=args.loss_type,
        classifier=args.classifier,
        lr_decay=args.lr_decay,
    )

    wandb_logger = WandbLogger(project=args.project_name, name=args.test_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_micro_f1",
        dirpath="./ckpt",
        filename="roberta-large-emb-lr_sched-{epoch:02d}-{val_micro_f1:.2f}",
        save_top_k=1,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    os.makedirs("./ckpt", exist_ok=True)
    trainer.fit(model=model, datamodule=dataloader)
