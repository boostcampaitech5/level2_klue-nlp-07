import pytorch_lightning as pl
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, LukeForEntityPairClassification
from loss import FocalLoss
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from utils import *
from emb_model import CustomRobertaForSequenceClassification
from functools import partial


class Model(pl.LightningModule):
    def __init__(
        self,
        model_name,
        lr=float(2e-5),
        num_labels=30,
        warmup_steps=500,
        warmup_ratio=-1.0,
        max_training_step=4500,
        loss_type="focal",
        classifier="default",
        lr_decay="default",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.max_training_step = max_training_step
        self.lr_decay = lr_decay

        model_config = AutoConfig.from_pretrained(model_name)
        model_config.num_labels = num_labels
        model_config.classifier = classifier

        # 사용할 모델을 호출합니다.
        # self.plm = AutoModelForSequenceClassification.from_pretrained(
        #     model_name, config=model_config
        # )
        if "luke" in model_name:
            self.plm = LukeForEntityPairClassification.from_pretrained(model_name, config=model_config)
        else:
            self.plm = CustomRobertaForSequenceClassification.from_pretrained(
            model_name, config=model_config
            )
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        if loss_type == "focal":
            self.loss_func = FocalLoss()
        elif loss_type == "label_smooth":
            self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
        elif loss_type == "cross_entropy":
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs):
        if "luke" in self.model_name:
            logits = self.plm(**inputs).logits
        else:
            logits = self.plm(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                entity_loc_ids=inputs["entity_loc_ids"],
            )
        # print(logits)

        # x = self.plm(x)[0]

        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch
        labels = batch["labels"]
        probs = self(inputs)
        preds = probs.argmax(-1)
        # print(preds)
        loss = self.loss_func(probs, labels.view(-1))
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
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
        x = batch
        labels = batch["labels"]
        probs = self(x)

        # self.log("test_auprc", klue_re_auprc(probs, labels))

    def predict_step(self, batch, batch_idx):
        inputs = batch
        logits = self(inputs)
        preds = logits.view(-1, 30).argmax(-1)
        result = {"preds": preds, "probs": logits}
        return result

    # 참고 자료 : https://lightning.ai/docs/pytorch/latest/common/optimization.html
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_exp_function(step: int, warmup_step: int, max_training_step: int):
            if step <= warmup_step:
                return step / warmup_step
            else:
                return 0.01 ** (
                    (step - warmup_step) / (max_training_step - warmup_step)
                )

        def lr_function(step: int, warmup_step: int, max_training_step: int):
            if step <= warmup_step:
                return step / warmup_step
            else:
                return 1 - (step - warmup_step) / (max_training_step - warmup_step)

        if self.lr_decay == "exp":
            function = lr_exp_function
        else:
            function = lr_function

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(
                function,
                warmup_step=self.warmup_steps,
                max_training_step=self.max_training_step,
            ),
        )

        return [optimizer], [
            {"scheduler": warmup_scheduler, "interval": "step", "name": "warmup+decay"}
        ]
