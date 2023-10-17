import argparse
import json
import os

import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from src.dataset import UFET
from src.utils import *
from src.config import Config



def bina(predict):
    max_index = torch.argmax(predict, dim=1)
    for dim, i in enumerate(max_index):
        predict[dim, i] = 0.51
    bina_predict = predict.gt(0.5).int()
    return bina_predict, predict


def record_metrics(step, label, predict):
    sample_num = label.size(0)
    # soft metric
    pos_acc = ((predict + label) == 2).sum().item() / label.sum().item()
    neg_acc = ((predict + label) == 0).sum().item() / (label == 0).sum().item()
    # strict metric: P==R==F1
    strict_acc = (torch.abs(predict - label).sum(1) == 0).sum().item() / label.size(0)

    # micro metric
    micro_p = (label * predict).sum() / predict.sum()
    micro_r = (label * predict).sum() / label.sum()
    micro_f1 = f1(micro_p, micro_r)

    # macro metric
    macro_p_list = []
    for t, p in zip((label * predict).sum(1), predict.sum(1)):
        if p.item() == 0:
            macro_p_list.append(0)
        else:
            macro_p_list.append(t.item() / p.item())
    macro_p = sum(macro_p_list) / len(macro_p_list)
    # macro_p = ((label * predict).sum(1) / predict.sum(1)).mean()
    macro_r = ((label * predict).sum(1) / label.sum(1)).mean()
    macro_f1 = f1(macro_p, macro_r)

    print('step %d\tmicro_f1: %f\tmacro_f1: %f\tstrict_acc: %f\tpos_acc: %f\tneg_acc: %f'
          % (step, micro_f1, macro_f1, strict_acc, pos_acc, neg_acc))
    print(f'step{step}: macro_P: {macro_p}, macro_R: {macro_r}, macro_F: {macro_f1}')
    return macro_p, macro_r, macro_f1, strict_acc


def record(model, data, step, device):
    model.eval()

    truth = []
    pos_predict = []
    bina_pos_predict = []
    neg_predict = []
    bina_neg_predict = []
    with torch.no_grad():
        for _, input_ids, attention_mask, labels in data:
            labels = labels.to(device)
            truth.append(labels)
            pos, neg = model(
                input_ids.to(device),
                attention_mask.to(device),
            )
            bina_pos_p, pos_p = bina(pos.sigmoid())
            pos_predict.append(pos_p)
            bina_pos_predict.append(bina_pos_p)

            bina_neg_p, neg_p = bina((-1 * neg).sigmoid())
            neg_predict.append(neg_p)
            bina_neg_predict.append(bina_neg_p)

        pos_predict = torch.cat(pos_predict, dim=0)
        bina_pos_predict = torch.cat(bina_pos_predict, dim=0)
        neg_predict = torch.cat(neg_predict, dim=0)
        bina_neg_predict = torch.cat(bina_neg_predict, dim=0)
        truth = torch.cat(truth, dim=0)

        print("****************  Positive  *********************")
        _, _, maf1, _ = record_metrics(step, truth, bina_pos_predict)
        # neg_truth = torch.cat(neg_truth, dim=0)
        print("****************  Negative  *********************")
        _, _, neg_maf1, _ = record_metrics(step, truth, bina_neg_predict)

        return pos_predict, neg_predict, truth


def main(args):
    device = "cuda:" + str(args.device)
    type2id = dict()
    id2type = dict()
    with open(args.ontology) as f:
        for line in f.readlines():
            type2id[line.strip()] = len(type2id)
            id2type[len(id2type)] = line.strip()

    # tokenizer = BertTokenizer.from_pretrained(args.backbone)
    tokenizer = BertTokenizer.from_pretrained(args.backbone)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})

    train_dataset = UFET(args.train, type2id)
    # test_dataset = UFET(args.test, type2id)
    test_train = DataLoader(
        dataset=train_dataset,
        batch_size=args.test_batch_size,
        collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
        drop_last=False,
        num_workers=0
    )
    # test = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=args.test_batch_size,
    #     shuffle=False,
    #     collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
    #     drop_last=False,
    #     num_workers=0
    # )

    model = torch.load(os.path.join(args.save_model_dir, 'model.pt')).to(device)
    # print("*********  On Test Dataset\n")
    # record(model, test, 1000, device)
    print("*********  On Train Dataset\n")
    if not os.path.exists(os.path.join(args.save_model_dir, 'pos_predict.pth')):
        pos_predict, neg_predict, truth = record(model, test_train, 1000, device)
        torch.save(pos_predict, os.path.join(args.save_model_dir, 'pos_predict.pth'))
        torch.save(neg_predict, os.path.join(args.save_model_dir, 'neg_predict.pth'))
        torch.save(truth, os.path.join(args.save_model_dir, 'truth.pth'))
    else:
        pos_predict = torch.load(os.path.join(args.save_model_dir, 'pos_predict.pth'))
        neg_predict = torch.load(os.path.join(args.save_model_dir, 'neg_predict.pth'))


    file_name = str(args.threshold) + '_train.txt'
    # file_name = 'wo_ina_train.txt'
    # file_name = 'wo_unl_train.txt'

    with open(args.train, 'r') as train, open(os.path.join(args.save_model_dir, file_name),'w') as denoise:
        train_lines = train.readlines()
        consistent = torch.abs(pos_predict - neg_predict).lt(args.threshold+0.01).int()  # 衡量差异

        both = (pos_predict.gt(0.49).int() + neg_predict.gt(0.49).int()).gt(0.99).int()
        for ins_id, train_line in enumerate(train_lines):
            train_line = json.loads(train_line.strip())
            predict = train_line['y_str']
            reduce = []

            label_idx = (both[ins_id] == 1.0).nonzero(as_tuple=False)
            for i in label_idx:
                if id2type[i.item()] not in predict:
                    predict.append(id2type[i.item()])

            for y in predict:
                if y in type2id and consistent[ins_id][type2id[y]] == 1.0:
                    reduce.append(y)

            # train_line['y_str'] = predict
            train_line['y_str'] = reduce

            if len(train_line['y_str']) != 0:
                denoise.write(json.dumps(train_line) + '\n')


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='./config/ultra.json')
        args = parser.parse_args()
        params = Config(args)

        params.save_model_dir = os.path.join(params.save_dir, params.dataset, params.mode, params.times)

        main(params)
    except Exception as e:
        logging.exception(e)
        raise
