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
        batch_size=32,
        shuffle=True,
        origin_train_path="../dataset/train/train_preprocessing.csv",  # 전처리만 한 origin trainset
        train_path="../dataset/train/train_split.csv",
        dev_path="../dataset/dev/dev.csv",
        test_path="../dataset/dev/dev.csv",
        predict_path="../dataset/test/test_data.csv",
        emb=True,
        use_stratified_kfold=False,  # StratifiedKFold 시엔 True
        train_indices=None,  # StratifiedKFold 시 필요한 trainset 인덱스
        val_indices=None,  # StratifiedKFold 시 필요한 valiset 인덱스
        only_entity=False,  # CoRE predict 시에만 사용
    ):
        super().__init__()
        # tokenizer 로드
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.origin_train_path = origin_train_path
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.emb = emb
        self.only_entity = only_entity

        self.use_stratified_kfold = use_stratified_kfold
        self.train_indices = train_indices
        self.val_indices = val_indices

    def preprocessing(self, data_path, stage):
        dataset = load_data(data_path)
        if stage == "predict" or stage == "test":
            label = []
        else:
            label = label_to_num(dataset["label"].values)
        if "luke" in self.model_name:
            dataset_token = luke_tokenized_dataset(dataset, self.tokenizer)
        else:
            dataset_token = tokenized_dataset(dataset, self.tokenizer, self.only_entity)
        return dataset_token, label

    # k-fold 시엔 즉석으로 dataset을 만들어서 사용하기 때문에 별도 함수 만들어 줍니다.
    def k_fold_preprocessing(self, dataset, stage):
        dataset = preprocessing_dataset(dataset)

        if stage == "predict":
            label = []
        else:
            label = label_to_num(dataset["label"].values)
        if "luke" in self.model_name:
            dataset_token = luke_tokenized_dataset(dataset, self.tokenizer)
        else:
            dataset_token = tokenized_dataset(dataset, self.tokenizer, self.only_entity)

        return dataset_token, label

    def setup(self, stage="fit"):
        if stage == "fit":
            # k-fold 시엔 즉석으로 dataset을 만들어서 사용하기 때문에 별도 처리 해준다.
            if self.use_stratified_kfold:
                # 학습데이터 준비
                origin_train_dataframe = pd.read_csv(self.origin_train_path)

                if self.train_indices is not None and self.val_indices is not None:
                    train_dataframe = origin_train_dataframe.iloc[self.train_indices]
                    val_dataframe = origin_train_dataframe.iloc[self.val_indices]

                # 학습데이터 준비
                train_inputs, train_targets = self.k_fold_preprocessing(
                    train_dataframe, "train"
                )

                # 검증데이터 준비
                val_inputs, val_targets = self.k_fold_preprocessing(
                    val_dataframe, "dev"
                )

                # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
                self.train_dataset = RE_Dataset(train_inputs, train_targets, "train")
                self.val_dataset = RE_Dataset(val_inputs, val_targets, "train")

            else:
                # 학습데이터 준비
                train_inputs, train_targets = self.preprocessing(
                    self.train_path, "train"
                )

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
