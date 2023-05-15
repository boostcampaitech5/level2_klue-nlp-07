from dataloader import Dataloader, RE_Dataset
import argparse
from model import Model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="klue/roberta-large", type=str)
    # parser.add_argument("--model_name", default="xlm-roberta-large", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epoch", default=10, type=int)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--origin_train_path", default="../dataset/train/train_preprocessing.csv")  # 전처리만 한 origin trainset
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
    parser.add_argument("--use_stratified_kfold", default=False)  # k-fold 사용 여부
    parser.add_argument("--num_folds", default=2) # k-fold의 분할 수
    args = parser.parse_args(args=[])



    # StratifiedKFold 사용할 경우
    if args.use_stratified_kfold:

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




        num_folds = args.num_folds
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42) # StratifiedKFold 객체 선언

        train_dataset = pd.read_csv(args.origin_train_path)
        indices = list(range(len(train_dataset)))        
        labels = train_dataset["label"].values
        folds = list(skf.split(indices, labels))  # labels 를 보고 분포에 맞게 skf 객체가 indices를 나눠준다 (train_index, val_index) 형태로 반환


        # k-fold 만큼 반복
        for fold, (train_index, val_index) in enumerate(folds):
            print(f"Training Fold {fold + 1}/{num_folds}")

            fold_dataloader = Dataloader(
                model_name=args.model_name,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                origin_train_path=args.origin_train_path,
                train_path=None, # StratifiedKFold 시엔 None
                dev_path=None, # StratifiedKFold 시엔 None
                test_path=None, # StratifiedKFold 시엔 None
                predict_path=None, # StratifiedKFold 시엔 None
                emb=args.emb,
                use_stratified_kfold=args.use_stratified_kfold,                                
                train_indices=train_index,
                val_indices=val_index,
            )

            trainer.fit(model=model, datamodule=fold_dataloader)


    
    else:

        dataloader = Dataloader(
            model_name=args.model_name,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            origin_train_path=None, # StratifiedKFold 시엔 None
            train_path=args.train_path,
            dev_path=args.dev_path,
            test_path=args.test_path,
            predict_path=args.predict_path,
            emb=args.emb,
            use_stratified_kfold=None, # StratifiedKFold 시엔 None
            train_indices=None, # StratifiedKFold 시엔 None
            val_indices=None, # StratifiedKFold 시엔 None
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