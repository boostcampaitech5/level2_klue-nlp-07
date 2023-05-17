import math
import multiprocessing
import pandas as pd
from datetime import datetime
from pytz import timezone
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForMaskedLM
from transformers import MLukeTokenizer, LukeForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import re
import sys
sys.path.append("..") # sibling 폴더 접근 위해
from baseline.utils import *
from tqdm import tqdm

SEED_SPLIT = 0
SEED_TRAIN = 42

MAX_SEQ_LEN = 256 # data max token len: 223(test) 231(train)
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 1e-5 
LR_WARMUP_STEPS = 100
LR_WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
EPOCHS = 1

TEST_SIZE = 0.15
MLM_PROB=0.15

now = datetime.now(timezone('Asia/Seoul')).strftime('%m-%d-%H:%M')
RUN_NAME = f"pretrainMLuke_[b{TRAIN_BATCH_SIZE},e{EPOCHS}]_{now}"
PROJECT_NAME = "huggingface"
MODEL_NAME = "studio-ousia/mluke-large"
OUTPUT_DIR = f"../model/{RUN_NAME}"

## data preparation
df1 = pd.read_csv("../dataset/test/test_data.csv")
df2 = pd.read_csv("../dataset/train/hoolly_train.csv")
df3 = pd.read_csv("../dataset/train/hoolly_dev.csv")
df = pd.concat([df1, df2, df3],ignore_index=True, axis=0)
df = df.drop(["id"], axis=1).reset_index().rename(columns={"index": "id"}) # 합친 dataframe index 순서대로
# df = df.set_index("id")
# df = df.rename(columns={"index": "id"})
# print("/n***dataframe***")
# print(df.iloc[0])

def origin_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    entity_sentence = []
    subject_entity_span = []
    object_entity_span = []
    for sen, i, j in zip(
        dataset["sentence"], dataset["subject_entity"], dataset["object_entity"]
    ):
        subj_start_idx = int(
            re.sub(r"[^0-9]", "", i[1:-1].split("'start_idx': ")[1][0:3])
        )
        subj_end_idx = int(re.sub(r"[^0-9]", "", i[1:-1].split("'end_idx': ")[1][0:3]))
        subject_entity_span.append((subj_start_idx, subj_end_idx+1))
        
        obj_start_idx = int(
            re.sub(r"[^0-9]", "", j[1:-1].split("'start_idx': ")[1][0:3])
        )
        obj_end_idx = int(re.sub(r"[^0-9]", "", j[1:-1].split("'end_idx': ")[1][0:3]))
        object_entity_span.append((obj_start_idx, obj_end_idx+1))
        
        i = i[1:-1].split(",")[0].split(":")[1]
        j = j[1:-1].split(",")[0].split(":")[1]

        subject_entity.append(i)
        object_entity.append(j)

    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "subject_entity_span": subject_entity_span,
            "object_entity": object_entity,
            "object_entity_span": object_entity_span,
            "label": dataset["label"],
        }
    )
    return out_dataset

origin_df = origin_dataset(df)
# entity_df = preprocessing_dataset(df)
# print("/n***orign_preprocessed***")
# print(origin_df.iloc[0])
# print("/n***entity_preprocessed***")
# print(entity_df.iloc[0])

# origin_df.to_csv("origin_df.csv")
# origin_df = origin_df[["id", "sentence", "subject_entity_span", "object_entity_span"]]
# origin_df = origin_df.set_index("id")
# print(origin_df.head(2))

### split to train and valid

df_train, df_valid = train_test_split(
    origin_df, test_size=TEST_SIZE, random_state=SEED_SPLIT
)

# print(df_train.iloc[0])

train_dataset = Dataset.from_pandas(df_train[["sentence", "subject_entity_span", "object_entity_span"]])
valid_dataset = Dataset.from_pandas(df_valid[["sentence", "subject_entity_span", "object_entity_span"]])

# print(valid_dataset[0])

# tokenizer and model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = MLukeTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME, use_fast=True, do_lower_case=False, max_len=MAX_SEQ_LEN
    )
model = LukeForMaskedLM.from_pretrained(MODEL_NAME)
model = model.to(device)

## dataset 토큰화
    
def luke_tokenize_function(dataset):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    entity_spans = []
    
    for subj_span, obj_span in zip(dataset["subject_entity_span"], dataset["object_entity_span"]):
        entity_spans.append([tuple(subj_span), tuple(obj_span)])
        # entity_spans.append([tuple(subj_span)])
        
    tokenized_sentences = tokenizer(
        list(dataset["sentence"]),
        entity_spans=entity_spans,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    return tokenized_sentences

column_names = train_dataset.column_names

train_dataset = train_dataset.map(
    luke_tokenize_function,
    batched=True,
    num_proc=multiprocessing.cpu_count(),
    remove_columns=column_names
)

valid_dataset = valid_dataset.map(
    luke_tokenize_function,
    batched=True,
    num_proc=multiprocessing.cpu_count(),
    remove_columns=column_names
)

print("\n**tokenized**")
# print(train_dataset[0])

## train
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB
)

# steps_per_epoch = int(len(train_dataset) / TRAIN_BATCH_SIZE)


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
    evaluation_strategy="no",
    # eval_steps=500,
    save_strategy="no",
    # save_steps=500,
    # save_total_limit=3,
    # load_best_model_at_end=False,
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,
    report_to="none",
    seed=SEED_TRAIN,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer
)

# wandb.init(project_name=PROJECT_NAME, run_name=RUN_NAME)

print("*** start training ***")
trainer.train()
trainer.save_model(OUTPUT_DIR)

### eval_loss 반환을 안 한다...
# my_eval_results = trainer.evaluate(eval_dataset=valid_dataset)
# print('My Evaluation results: ', my_eval_results)
# print(f"Perplexity: {math.exp(my_eval_results['eval_loss']):.3f}")

# Evaluate the model
print("***start evaluation***")
model.eval()

cnt = 0
eval_loss = 0.0
with torch.no_grad():
    for batch in tqdm(trainer.get_eval_dataloader()):
        batch = batch.to(device)
        inputs = trainer._prepare_inputs(batch)
        outputs = trainer.model(**inputs)
        x = outputs.loss.item()
        eval_loss += x
        cnt += 1

# num_eval_samples = len(valid_dataset)
eval_loss /= cnt

print(f"Evaluation loss: {eval_loss:.5f}")
print(f"Perplexity: {math.exp(eval_loss):.5f}")
