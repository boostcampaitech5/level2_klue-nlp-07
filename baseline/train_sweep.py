from dataloader import Dataloader, RE_Dataset
import argparse
from model import Model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from omegaconf import OmegaConf
from transformers import set_seed
import math


if __name__ == "__main__":
    print("*" * 100)
    print("train_sweep.py")
    print("*" * 100)

    conf = OmegaConf.load("./config.yaml")

    # seed 고정
    if conf.params.seed > 0:
        set_seed(conf.params.seed)

    # k-fold 여부에 따라 step 수가 달라짐.
    if conf.params.use_stratified_kfold:
        train_df = pd.read_csv(conf.path.origin_train_path)  # 학습 데이터

        # ((num_folds-1) x num_folds) 를 곱해줘야 함.
        # 총 스텝 수 = 학습 데이터 개수 / 배치사이즈 x max_epoch / ((num_folds-1) x num_folds)
        total_steps = math.ceil(
            (len(train_df) // conf.params.batch_size)
            * conf.params.max_epoch
            * ((conf.params.num_folds - 1) / conf.params.num_folds)
        )

    else:
        train_df = pd.read_csv(conf.path.train_path)  # 학습 데이터
        # 총 스텝 수 = 학습 데이터 개수 / 배치사이즈 x max_epoch
        total_steps = (len(train_df) // conf.params.batch_size) * conf.params.max_epoch

    # warmup_ratio 가 음수가 아닌 경우에만 warmup_staps 를 overwrite 합니다.
    if conf.params.warmup_ratio > 0.0:  # float라서 등호(=) 주의
        conf.params.warmup_steps = int(total_steps * conf.params.warmup_ratio)

    # ----------------- sweep -----------------
    sweep_conf = OmegaConf.load("./sweep.yaml")

    # sweep.yml 의 인자를 저장할 set
    update_config_set = set()

    for key, value in sweep_conf.items():
        if "parameters" == key:
            for sub_key, sub_value in value.items():
                update_config_set.add(sub_key)

    # parser 를 통해 sweep이 전달하는 cli 를 받아오도록 처리
    parser = argparse.ArgumentParser()

    # update_config_set 에 대해서만 arg를 만들어준다.
    for arg in update_config_set:
        if arg in ["model_name", "loss_type", "classifier", "lr_decay"]:
            parser.add_argument(f"--{arg}", type=str, default=None)
        elif arg in [
            "batch_size",
            "max_epoch",
            "num_labels",
            "warmup_steps",
            "num_folds",
            "seed",
        ]:
            parser.add_argument(f"--{arg}", type=int, default=None)
        elif arg in ["learning_rate", "weight_decay", "warmup_ratio"]:
            parser.add_argument(f"--{arg}", type=float, default=None)
        elif arg in ["shuffle", "emb"]:
            parser.add_argument(f"--{arg}", type=bool, default=None)

    args = parser.parse_args()

    # 기존 conf에 args 덮어씌움으로써 sweep의 내용들이 전달 될 수 있도록 처리
    for arg in update_config_set:
        if getattr(args, arg) is not None:
            print(getattr(args, arg))

            if "model_name" == arg:
                conf.model_name = getattr(args, arg)
            else:
                conf.params.__setattr__(arg, getattr(args, arg))

    # ----------------- sweep -----------------

    model = Model(
        model_name=conf.model_name,
        lr=conf.params.learning_rate,
        num_labels=conf.params.num_labels,
        warmup_steps=conf.params.warmup_steps,
        max_training_step=total_steps,
        loss_type=conf.params.loss_type,
        classifier=conf.params.classifier,
        lr_decay=conf.params.lr_decay,
    )

    wandb_logger = WandbLogger(
        project=conf.params.project_name, name=conf.params.test_name, config=conf
    )

    # checkpoint_callback = ModelCheckpoint(
    #         monitor="val_micro_f1",
    #         dirpath="./ckpt",
    #         filename="roberta-large-emb-lr_sched-{epoch:02d}-{val_micro_f1:.2f}",
    #         save_top_k=0,
    #         mode="max",
    #     )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    os.makedirs("./ckpt", exist_ok=True)

    # StratifiedKFold 사용할 경우
    if conf.params.use_stratified_kfold:
        num_folds = conf.params.num_folds
        skf = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=42
        )  # StratifiedKFold 객체 선언

        train_dataset = pd.read_csv(conf.path.origin_train_path)
        indices = list(range(len(train_dataset)))
        labels = train_dataset["label"].values
        folds = list(
            skf.split(indices, labels)
        )  # labels 를 보고 분포에 맞게 skf 객체가 indices를 나눠준다 (train_index, val_index) 형태로 반환

        # k-fold 만큼 반복
        for fold, (train_index, val_index) in enumerate(folds):
            print(f"Training Fold {fold + 1}/{num_folds}")

            # fold 별로 trainer, dataloader 객체를 새로 만들어준다
            trainer = pl.Trainer(
                accelerator="gpu",
                max_epochs=conf.params.max_epoch,
                log_every_n_steps=1,
                logger=wandb_logger,
                callbacks=[lr_monitor],
                enable_checkpointing=False,
                precision="16-mixed",
            )

            fold_dataloader = Dataloader(
                model_name=conf.model_name,
                batch_size=conf.params.batch_size,
                shuffle=conf.params.shuffle,
                origin_train_path=conf.path.origin_train_path,
                train_path=None,  # StratifiedKFold 시엔 None
                dev_path=None,  # StratifiedKFold 시엔 None
                test_path=None,  # StratifiedKFold 시엔 None
                predict_path=None,  # StratifiedKFold 시엔 None
                emb=conf.params.emb,
                use_stratified_kfold=conf.params.use_stratified_kfold,
                train_indices=train_index,
                val_indices=val_index,
            )

            trainer.fit(model=model, datamodule=fold_dataloader)

    else:
        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=conf.params.max_epoch,
            log_every_n_steps=1,
            logger=wandb_logger,
            callbacks=[lr_monitor],
            enable_checkpointing=False,
            precision="16-mixed",
        )

        dataloader = Dataloader(
            model_name=conf.model_name,
            batch_size=conf.params.batch_size,
            shuffle=conf.params.shuffle,
            origin_train_path=None,  # StratifiedKFold 시엔 None
            train_path=conf.path.train_path,
            dev_path=conf.path.dev_path,
            test_path=conf.path.test_path,
            predict_path=conf.path.predict_path,
            emb=conf.params.emb,
            use_stratified_kfold=None,  # StratifiedKFold 시엔 None
            train_indices=None,  # StratifiedKFold 시엔 None
            val_indices=None,  # StratifiedKFold 시엔 None
        )

        trainer.fit(model=model, datamodule=dataloader)
