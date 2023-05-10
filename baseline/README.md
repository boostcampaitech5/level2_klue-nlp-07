## 파일 구조

### ❗️**파일 구조에 맞춰서 파일들을 배치해주세요❗️**

baseline 폴더와 dataset 폴더는 다른 디렉토리에 있음에 주의해주세요.

```
📦pl_refactoring
 ┣ 📂baseline
 ┃ ┣ 📂ckpt
 ┃ ┃ ┗ 📜roberta-large-epochepoch=03-val_micro_f1val_micro_f1=86.25.ckpt
 ┃ ┣ 📂submission
 ┃ ┃ ┗ 📜sample_submission.csv
 ┃ ┣ 📜dataloader.py
 ┃ ┣ 📜dict_label_to_num.pkl
 ┃ ┣ 📜dict_num_to_label.pkl
 ┃ ┣ 📜loss.py
 ┃ ┣ 📜model.py
 ┃ ┣ 📜predict.py
 ┃ ┣ 📜train.py
 ┃ ┗ 📜utils.py
 ┣ 📂dataset
 ┃ ┣ 📂dev
 ┃ ┃ ┗ 📜dev.csv
 ┃ ┣ 📂test
 ┃ ┃ ┗ 📜test_data.csv
 ┃ ┣ 📂train
 ┃ ┃ ┣ 📜train.csv
 ┃ ┃ ┗ 📜train_split.csv
```

# 사용법

## 학습

```python
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
args = parser.parse_args(args=[])
```

모든 파라미터들은 argument로 관리된다고 생각하시면 됩니다. 추후 config.yaml로 관리하게끔 리팩토링하기 쉽게 모아두었습니다.

사용하고 싶은 모델에 따라서 model_name을 수정하시거나, 배치 사이즈, 에폭 등을 수정할 수 있습니다.

`project_name`과 `test_name`은 wandb 기록용으로 두었습니다.

warmup_steps는 아직 에러가 있어 수정 중입니다. 무시하셔도 됩니다.

loss_type은 `focal`과 `label_smooth`, `cross_entropy`가 있습니다. 기호에 따라 사용하시면 됩니다.

학습 코드를 실행할 땐 꼭 `baseline` 디렉토리로 이동한 후, `python train.py`를 실행해주세요.

(안 그러면 데이터 경로 오류가 납니다,,!!)

현재 콜백은 val_micro_f1 점수가 가장 높은 모델 하나만 저장하게끔 해뒀고,

용도에 따라서 earlystopping 콜백을 추가하시거나 checkpoint callback을 수정하셔도 될 것 같습니다.

## 추론

학습 코드를 돌리면 `./ckpt` 경로에 train.py의 checkpoint_callback에서 정해둔 filename의 양식으로 체크포인트가 저장됩니다. 

(ex. ‘./ckpt/roberta-large-epochepoch=03-val_micro_f1val_micro_f1=86.25.ckpt’)

추론 코드에서 checkpoint를 불러오는 부분이 있는데, 그 경로에 체크포인트의 경로를 넣어주시면 됩니다.

또한, 추론 코드를 돌리기 전에 `./submission` 경로에 베이스라인에서 주어진 `sample_submission.csv` 파일을 넣어두어야 합니다. 샘플을 불러와서 값을 수정하고 다른 csv로 저장하는 식으로 코드를 짰습니다.

# 특이사항

- 측정 metric들과 processing_dataset, load_data 등 메소드들은 utils.py에 모두 놔뒀습니다.
- auprc는 아직 추가하지 못했습니다. 추후 추가할 예정입니다.
- pkl 파일도 혹시 몰라서 깃허브에 업로드 하지 않았습니다. 꼭 baseline 디렉토리에 추가해주세요.
