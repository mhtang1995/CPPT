import argparse
import os
import time
from datetime import timedelta

import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers.optimization import AdamW
from torch.utils.data import DataLoader
from src.model import PTuning, step_two_loss, basic_loss
from src.dataset import UFET
from src.utils import *
from src.config import Config



def evaluation(model, data, step, device):
    model.eval()
    truth = []
    predict = []
    with torch.no_grad():
        for _, input_ids, attention_mask, labels in data:
            labels = labels.to(device)
            truth.append(labels)

            p, _ = model(
                input_ids.to(device),
                attention_mask.to(device),
            )
            predict.append(binarization(p.sigmoid()))

        predict = torch.cat(predict, dim=0)
        truth = torch.cat(truth, dim=0)
        logging.info("****************  Positive  *********************")
        _, _, maf1, _ = record_metrics(step, truth, predict)

        return maf1


def main(args):
    args.train = os.path.join(params.save_model_dir, str(args.threshold) + '_train.txt')
    # args.train = os.path.join(params.save_model_dir, 'wo_unl_train.txt')
    # args.train = os.path.join(params.save_model_dir, 'wo_ina_train.txt')

    device = "cuda:" + str(args.device)
    type2id = dict()
    id2type = dict()
    with open(args.ontology) as f:
        for line in f.readlines():
            type2id[line.strip()] = len(type2id)
            id2type[len(id2type)] = line.strip()

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.backbone)
    mask_id = tokenizer.mask_token_id
    # add token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
    prompt_placeholder_id = tokenizer.additional_special_tokens_ids[0]
    unk_id = tokenizer.unk_token_id

    train_dataset = UFET(args.train, type2id)
    dev_dataset = UFET(args.valid, type2id)
    test_dataset = UFET(args.test, type2id)

    # create dataset
    train = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
        drop_last=False,
        num_workers=0
    )
    valid = DataLoader(
        dataset=dev_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
        drop_last=False,
        num_workers=0
    )
    test = DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
        drop_last=False,
        num_workers=0
    )

    model = PTuning(
        args=args,
        type_num=len(type2id),
        prompt_placeholder_id=prompt_placeholder_id,
        unk_id=unk_id,
        mask_id=mask_id,
        backbone=args.backbone,
        embedding_dim=args.bert_size,
        dense_param=os.path.join(args.save_dir, args.bert_name, args.dense_param),
        ln_param=os.path.join(args.save_dir, args.bert_name, args.ln_param),
        fc_param=os.path.join(args.save_dir, args.bert_name, args.fc_param)
    )
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # metric
    step = 0
    best_step = 0
    best_f1 = 0
    kill = 0
    early_stop = False
    log = {
        'loss': {
            '1': []
        }
    }

    # training loop
    for epoch in range(args.max_epoch):
        for idx, input_ids, attention_mask, labels in train:
            model.train()
            step += 1
            predict1, predict2 = model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))


            loss, pos_loss = step_two_loss(
                predict=predict1,
                labels=labels.to(device),
                criterion=criterion)

            log["loss"]["1"].append(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_step == 0:
                logging.info(f'step{step}'
                             f' avg_pos_loss:{sum([_ for _ in log["loss"]["1"]]) / len(log["loss"]["1"])}')
                log = {
                    'loss': {
                        '1': []
                    }
                }
            # valid
            if step % args.valid_step == 0:
                logging.info("*********  On DEV Dataset\n")
                f1 = evaluation(model, test, step, device)

                if f1 > best_f1:
                    torch.save(model, os.path.join(args.save_model_dir, str(params.threshold) + 'clean'))
                    best_step = step
                    kill = 0
                    best_f1 = f1
                else:
                    kill += 1

                if kill >= args.patience:
                    early_stop = True
                    break
        if early_stop:
            break

    # test
    model = torch.load(os.path.join(args.save_model_dir, str(params.threshold) + 'clean')).to(device)
    evaluation(model, test, step, device)
    logging.info("Best step is: " + str(best_step) + " best f1 is: " + str(best_f1.item()))
    torch.cuda.empty_cache()



if __name__ == '__main__':
    try:

        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='./config/ultra.json')
        args = parser.parse_args()
        params = Config(args)
        params.save_model_dir = os.path.join(params.save_dir, params.dataset, params.mode, params.times)

        set_logger(os.path.join(params.save_model_dir, 'clean.log'))
        main(params)
    except Exception as e:
        logging.exception(e)
        raise
