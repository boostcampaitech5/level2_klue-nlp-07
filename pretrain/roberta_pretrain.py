import math
import multiprocessing
import pandas as pd
import wandb
from datetime import datetime
from pytz import timezone
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import torch

SEED_SPLIT = 0
SEED_TRAIN = 42

MAX_SEQ_LEN = 235 # data max token len: 223(test) 231(train)
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5 
LR_WARMUP_STEPS = 100
LR_WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
EPOCHS = 1

TEST_SIZE = 0.15
MLM_PROB=0.15

now = datetime.now(timezone('Asia/Seoul')).strftime('%m-%d-%H:%M')
RUN_NAME = f"pretrainLen225_[b{TRAIN_BATCH_SIZE},e{EPOCHS}]_{now}"
PROJECT_NAME = "huggingface"
MODEL_NAME = "klue/roberta-large"
OUTPUT_DIR = "./results/pretrain_roberta_test10"

## data preparation

df1 = pd.read_csv("../dataset/test/test_data.csv")
df2 = pd.read_csv("../dataset/train/origin_train.csv")
df = pd.concat([df1, df2],ignore_index=True, axis=0)
df = df.drop(["id"], axis=1).reset_index().rename({"index": "id"}) # 합친 dataframe index 순서대로

df_train, df_valid = train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED_SPLIT
)

train_dataset = Dataset.from_pandas(df_train[['sentence']].dropna())
valid_dataset = Dataset.from_pandas(df_valid[['sentence']].dropna())

## tokenizer and model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME, use_fast=True, do_lower_case=False, max_len=MAX_SEQ_LEN
    )
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model = model.to(device)

## dataset 토큰화

def tokenize_function(row):
    # row: datset 전체
    return tokenizer(
        row["sentence"],
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_special_tokens_mask=True
    )

column_names = train_dataset.column_names

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=multiprocessing.cpu_count(),
    remove_columns=column_names
)

valid_dataset = valid_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=multiprocessing.cpu_count(),
    remove_columns=column_names
)

## Trainer 설정

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB
)

# steps_per_epoch = int(len(train_dataset) / TRAIN_BATCH_SIZE)


wandb.init(project=PROJECT_NAME, name=RUN_NAME)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir="./logs/pretrain_roberta",
    num_train_epochs=EPOCHS,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    # warmup_steps=LR_WARMUP_STEPS,
    warmup_ratio=LR_WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="no",
    # save_steps=500,
    # save_total_limit=3,
    # load_best_model_at_end=False,
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,
    report_to="wandb",
    seed=SEED_TRAIN
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer
)

## 학습

trainer.train()
trainer.save_model(OUTPUT_DIR)

my_eval_results = trainer.evaluate()
print('My Evaluation results: ', my_eval_results)
print(f"Perplexity: {math.exp(my_eval_results['eval_loss']):.3f}")
