import pytorch_lightning as pl
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from loss import FocalLoss
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from utils import *


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, num_labels, warmup_steps, loss_type):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.warmup_steps = warmup_steps

        model_config = AutoConfig.from_pretrained(model_name)
        model_config.num_labels = num_labels

        # 사용할 모델을 호출합니다.
        self.plm = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=model_config
        )
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        if loss_type == "focal":
            self.loss_func = FocalLoss()
        elif loss_type == "label_smooth":
            self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
        elif loss_type == "cross_entropy":
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.plm(x)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        inputs = batch["input_ids"]
        labels = batch["labels"]
        probs = self(inputs)

        loss = self.loss_func(probs.view(-1, 30), labels.view(-1))
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["input_ids"]
        labels = batch["labels"]
        logits = self(inputs)

        preds = logits.view(-1, 30).argmax(-1)
        preds_ = preds.clone().detach().cpu().numpy()
        labels_ = labels.clone().detach().cpu().numpy()
        micro_f1 = klue_re_micro_f1(preds_, labels_)

        loss = self.loss_func(logits.view(-1, 30), labels.view(-1))
        self.log("val_loss", loss)
        self.log("val_micro_f1", micro_f1)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch["input_ids"]
        labels = batch["labels"]
        probs = self(x)

        # self.log("test_auprc", klue_re_auprc(probs, labels))

    def predict_step(self, batch, batch_idx):
        inputs = batch["input_ids"]
        logits = self(inputs)
        preds = logits.view(-1, 30).argmax(-1)
        result = {"preds": preds, "probs": logits}
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        #         def warmup_scheduler(step):
        #             warmup_factor = min(1.0, step / self.warmup_steps)
        #             return warmup_factor * self.lr

        #         scheduler = LambdaLR(optimizer, lr_lambda=warmup_scheduler)

        #         return [optimizer], [scheduler]
        return optimizer
