from dataloader import Dataloader
import argparse
from model import Model
import pytorch_lightning as pl
import torch
from utils import num_to_label
import pandas as pd
import torch.nn.functional as F


if __name__ == "__main__":
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
        result_resoftmax=None,  # subject entity에 따른 결과 보정
    )

    model = Model.load_from_checkpoint(
        "./ckpt/roberta-large-emb-lr_sched(exp)-epoch=05-val_micro_f1=86.56.ckpt"
    )

    trainer = pl.Trainer(
        accelerator="gpu", max_epochs=args.max_epoch, log_every_n_steps=1, logger=False
    )

    prediction = trainer.predict(model=model, datamodule=dataloader)

    # preds = []
    # probs = []
    # for item in prediction:
    #     pred = item["preds"].tolist()
    #     prob = item["probs"]
    #     preds += pred
    #     probs += prob
    # preds = num_to_label(preds)
    # probs = [F.softmax(prob, dim=-1).tolist() for prob in probs]
    
    def softmax(a) :
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
    return y

    test = pd.read_csv("../dataset/test/test_data.csv")
    
    subject_type = [] #test dataset의 subject entity를 순서대로 append
    for i in range(len(test)):
        subject_type.append(eval(test.loc[i, 'subject_entity'])['type'])
    
    new_prob = []
    new_pred = []
    for i in range(len(subject_type)):
        temp = []
        item = prediction[i]
        pred = item["preds"].tolist()
        prob = item["probs"]
        if subject_type[i] == 'PER':
            softmax_list = []
            
            for p in range(30):
                if p in [4,5,6,10,11,12,13,14,15,16,17,21,23,24,25,26,27,29]: #per일 때
                    softmax_list.append(prob[p])
            softmax_list = softmax(softmax_list)
            pp = 0
            for p in range(30):
                if p in [1,2,3,5,7,9,18,19,20,22,28]:
                    temp.append(0)
                else:
                    temp.append(softmax_list[pp])
                    pp+=1
        else: #org
            softmax_list = []
            
            for p in range(30):
                if p in [1,2,3,5,7,9,18,19,20,22,28]: #org일 때
                    softmax_list.append(prob[p])
            softmax_list = softmax(softmax_list)
            pp = 0
            for p in range(30):
                if p in [4,5,6,10,11,12,13,14,15,16,17,21,23,24,25,26,27,29]:
                    temp.append(0)
                else:
                    temp.append(softmax_list[pp])
                    pp+=1
        new_prob.append(str(temp))
        new_pred.append(pred)
    preds = num_to_label(new_pred)
    
    

    submission = pd.read_csv("./submission/sample_submission.csv")
    submission["pred_label"] = preds
    submission["probs"] = new_prob #probs
    submission.to_csv("./submission/submission.csv", index=False)
