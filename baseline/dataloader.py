import torch
import pytorch_lightning as pl
from utils import *
from transformers import AutoTokenizer


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels, stage):
        self.pair_dataset = pair_dataset
        self.labels = labels
        self.stage = stage

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.pair_dataset.items()
        }
        if self.stage != "predict":
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.pair_dataset["input_ids"])


class Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        shuffle,
        train_path,
        dev_path,
        test_path,
        predict_path,
        emb,
    ):
        super().__init__()
        # tokenizer 로드
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.emb = emb

    def preprocessing(self, data_path, stage):
        dataset = load_data(data_path)
        if stage == "predict":
            label = []
        else:
            label = label_to_num(dataset["label"].values)
        if self.emb:
            dataset_token = emb_tokenized_dataset(dataset, self.tokenizer)
        else:
            dataset_token = tokenized_dataset(dataset, self.tokenizer)
        return dataset_token, label

    def setup(self, stage="fit"):
        if stage == "fit":
            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(self.train_path, "train")

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(self.dev_path, "dev")

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = RE_Dataset(train_inputs, train_targets, "train")
            self.val_dataset = RE_Dataset(val_inputs, val_targets, "train")
        else:
            # 평가데이터 준비
            test_inputs, test_targets = self.preprocessing(self.test_path, "test")
            self.test_dataset = RE_Dataset(test_inputs, test_targets, "train")

            predict_inputs, predict_targets = self.preprocessing(
                self.predict_path, "predict"
            )
            self.predict_dataset = RE_Dataset(predict_inputs, [], "predict")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.batch_size
        )
