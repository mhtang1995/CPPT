
import argparse
import os

import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers.optimization import AdamW
from torch.utils.data import DataLoader
from src.model import PTuning, step_one_loss, basic_loss
from src.dataset import UFET
from src.utils import *
from src.config import Config




def evaluation(model, data, step, device, is_train=False):
    model.eval()
    truth = []
    predict1 = []
    predict2 = []
    with torch.no_grad():
        for _, input_ids, attention_mask, labels in data:
            labels = labels.to(device)
            truth.append(labels)

            p1, p2 = model(
                input_ids.to(device),
                attention_mask.to(device),
            )
            predict1.append(binarization(p1.sigmoid()))
            predict2.append(binarization((-1 * p2).sigmoid()))

        predict1 = torch.cat(predict1, dim=0)
        predict2 = torch.cat(predict2, dim=0)
        truth = torch.cat(truth, dim=0)

        logging.info("****************  Positive  *********************")
        _, _, p1f1, _ = record_metrics(step, truth, predict1)
        logging.info("****************  Negative  *********************")
        _, _, p2f1, _ = record_metrics(step, truth, predict2)

        if is_train:
            inconsistent_num = torch.abs(predict1 - predict2).sum().item()
            rate = inconsistent_num / (predict1.size(0) * predict1.size(1))
            logging.info(f"Divergent Co-Prediction: {step} {rate}")


        return p1f1, p2f1



def main(args):
    device = "cuda:" + str(args.device)
    # Read ontology
    type2id = dict()
    id2type = dict()
    with open(args.ontology) as f:
        for line in f.readlines():
            type2id[line.strip()] = len(type2id)
            id2type[len(id2type)] = line.strip()

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.backbone)
    mask_id = tokenizer.mask_token_id
    # add token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
    prompt_placeholder_id = tokenizer.additional_special_tokens_ids[0]
    unk_id = tokenizer.unk_token_id

    # Dataset
    train_dataset = UFET(args.train, type2id)
    dev_dataset = UFET(args.valid, type2id)
    test_dataset = UFET(args.test, type2id)

    train = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True,
                       collate_fn=lambda x: UFET.collate_fn(x, tokenizer), drop_last=False, num_workers=0)
    test_train = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=False,
                            collate_fn=lambda x: UFET.collate_fn(x, tokenizer), drop_last=False, num_workers=0)
    valid = DataLoader(dataset=dev_dataset, batch_size=args.test_batch_size, shuffle=False,
                       collate_fn=lambda x: UFET.collate_fn(x, tokenizer), drop_last=False, num_workers=0)
    test = DataLoader(dataset=test_dataset,batch_size=args.test_batch_size, shuffle=False,
                      collate_fn=lambda x: UFET.collate_fn(x, tokenizer), drop_last=False, num_workers=0)

    # Model
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
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # metric
    step = 0
    best_step = 0
    best_f1 = [0, 0, 0]
    kill = 0
    early_stop = False
    log = {
        'loss': {
            '1': [],
            '2': []
        }
    }

    # training loop
    for epoch in range(args.max_epoch):
        for idx, input_ids, attention_mask, labels in train:
            step += 1

            model.train()
            predict1, predict2 = model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))

            loss, loss1, loss2 = step_one_loss(
            # loss, loss1, loss2 = basic_loss(
                args=args,
                predict1=predict1,
                predict2=predict2,
                labels=labels.to(device),
                criterion=criterion
            )

            log["loss"]["1"].append(loss1)
            log["loss"]["2"].append(loss2)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_step == 0:
                logging.info(f'step{step}'
                             f' avg_pos_loss:{sum([_ for _ in log["loss"]["1"]]) / len(log["loss"]["1"])}'
                             f' avg_neg_loss:{sum([_ for _ in log["loss"]["2"]]) / len(log["loss"]["2"])}')
                log = {
                    'loss': {
                        '1': [],
                        '2': [],
                    }
                }

            # valid
            if step % args.valid_step == 0:
                logging.info("*********  On DEV Dataset\n")
                f1 = evaluation(model, valid, step, device)
                # evaluation(model, test_train, step, device, is_train=True)

                if f1[0] > best_f1[0]:
                    torch.save(model, os.path.join(args.save_model_dir, 'model.pt'))
                    kill = 0
                    best_f1 = f1
                    best_step = step
                else:
                    kill += 1

                if kill >= args.patience:
                    early_stop = True
                    break
        if early_stop:
            break
    logging.info("Best step is: " + str(best_step) + " best f1 is: " + str(best_f1[0].item()))
    # test
    model = torch.load(os.path.join(args.save_model_dir, 'model.pt')).to(device)
    evaluation(model, test, step, device)
    # logging.info("*********  On Train Dataset\n")
    # evaluation(model, test_train, step, device)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='./config/ultra.json')
        args = parser.parse_args()
        params = Config(args)

        params.save_model_dir = os.path.join(params.save_dir, params.dataset, params.mode, params.times)
        if not os.path.exists(params.save_model_dir):
            os.makedirs(params.save_model_dir)
        set_logger(os.path.join(params.save_model_dir, 'log.txt'))

        main(params)
    except Exception as e:
        logging.exception(e)
        raise
