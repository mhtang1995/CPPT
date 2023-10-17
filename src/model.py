import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class PTuning(nn.Module):
    def __init__(self,
                 args,
                 type_num,
                 prompt_placeholder_id,
                 unk_id,
                 mask_id,
                 backbone,
                 embedding_dim,
                 dense_param,
                 ln_param,
                 fc_param,
                 init_template=None
                 ):
        super(PTuning, self).__init__()

        self.bert = BertModel.from_pretrained(backbone)

        self.embedding_dim = embedding_dim
        self.type_num = type_num

        self.mask_split = args.mask_split
        self.mask_id = mask_id

        self.prompt_encoder = Prompt(
            3, self.embedding_dim, self.bert.get_input_embeddings(), prompt_placeholder_id, unk_id, init_template)

        self.mask_encoder = Mask(self.embedding_dim, self.bert.get_input_embeddings(), self.mask_id)

        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.ln = nn.LayerNorm(self.embedding_dim, elementwise_affine=True)
        self.fc = nn.Linear(self.embedding_dim, type_num, bias=True)
        self.init_weight(dense_param, ln_param, fc_param)

    def init_weight(self, dense_param, ln_param, fc_param):
        self.dense.load_state_dict(torch.load(dense_param))
        self.ln.load_state_dict(torch.load(ln_param))
        self.fc.load_state_dict(torch.load(fc_param))


    def encoder(self, input_ids, attention_mask):
        # 处理 [PROMPT] [MASK]
        input_embeds = self.prompt_encoder(input_ids, self.bert.get_input_embeddings())
        if self.mask_split:
            input_embeds = self.mask_encoder(input_ids, input_embeds)
        output = self.bert(inputs_embeds=input_embeds, attention_mask=attention_mask)[0]

        _, mask_idx = (input_ids == self.mask_id).nonzero(as_tuple=True)
        mask_idx = mask_idx.reshape(input_ids.shape[0], -1)
        mask1 = []
        mask2 = []

        for i in range(input_ids.shape[0]):
            mask1.append(output[i][mask_idx[i][0]])
            mask2.append(output[i][mask_idx[i][1]])
        mask1 = torch.stack(mask1)
        mask2 = torch.stack(mask2)
        return mask1, mask2


    def forward(self, input_ids, attention_mask):
        mask_embeddings = self.encoder(input_ids, attention_mask)
        # [MASK] predict
        predict1 = self.fc(self.ln(self.dense(mask_embeddings[0])))
        predict2 = self.fc(self.ln(self.dense(mask_embeddings[1])))

        return predict1, predict2






def basic_loss(args, predict1, predict2, labels, criterion):
    loss1 = criterion(predict1, labels)
    loss2 = criterion(predict2, 1 - labels)
    loss = 0.5 * (loss1 + loss2).mean()
    # loss = loss1.mean()

    return loss, loss1.mean().item(), loss2.mean().item()


def step_one_loss(args, predict1, predict2, labels, criterion):

    loss1 = criterion(predict1, labels)
    loss2 = criterion(predict2, 1 - labels)
    co_loss = 0.5 * (loss1 + loss2)

    p1 = predict1.sigmoid().gt(0.5).int()
    p2 = (-1 * predict2).sigmoid().gt(0.5).int()
    inconsistent = torch.abs(p1 - p2)
    consistent = 1 - inconsistent

    if p1.sum().item() == 0:
        f1 = 0
    else:
        p = (labels * p1).sum() / p1.sum()
        r = (labels * p2).sum() / labels.sum()
        if (p + r).item() == 0:
            f1 = 0
        else:
            f1 = (2 * p * r / (p + r)).item()

    if f1 <= args.threshold_one:
        loss = co_loss.mean()
    else:
        if inconsistent.sum().item() != 0:
            if args.loss_version == 1:
                loss = (args.weight * inconsistent * co_loss).sum() / inconsistent.sum() \
                       + (consistent * co_loss).sum() / consistent.sum()
            else:
                loss = (args.weight * inconsistent * co_loss + consistent * co_loss).mean()
        else:
            loss = (consistent * co_loss).sum() / consistent.sum()


    return loss, loss1.mean().item(), loss2.mean().item()



def step_two_loss(predict, labels, criterion):
    loss = criterion(predict, labels)
    return loss.mean(), loss.mean().item()






class Prompt(nn.Module):

    def __init__(self, prompt_length, embedding_dim, bert_embedding, prompt_placeholder_id, unk_id, init_template=None):
        super(Prompt, self).__init__()
        self.prompt_length = prompt_length
        self.prompt_placeholder_id = prompt_placeholder_id
        self.unk_id = unk_id
        self.prompt = nn.Parameter(torch.randn(self.prompt_length, embedding_dim))

        if init_template is not None:
            print(init_template)
            self.prompt = nn.Parameter(bert_embedding(init_template).clone())

    def forward(self, input_ids, bert_embedding):
        bz = input_ids.shape[0]

        raw_embedding = input_ids.clone()
        raw_embedding[raw_embedding == self.prompt_placeholder_id] = self.unk_id
        raw_embedding = bert_embedding(raw_embedding)  # (bz, len, embedding_dim)
        if self.prompt_length != 0:
            prompt_idx = torch.nonzero(input_ids == self.prompt_placeholder_id, as_tuple=False)
            prompt_idx = prompt_idx.reshape(bz, self.prompt_length, -1)[:, :, 1]    # (bz, prompt_len)
            for b in range(bz):
                for i in range(self.prompt_length):
                    raw_embedding[b, prompt_idx[b, i], :] = self.prompt[i, :]
        return raw_embedding


class Mask(nn.Module):
    def __init__(self, embedding_dim, bert_embedding, mask_id):
        super(Mask, self).__init__()
        self.embedding_dim = embedding_dim
        self.mask_id = mask_id
        self.mask = nn.Parameter(bert_embedding(torch.tensor([mask_id] * 3)).clone())

    def forward(self, input_ids, input_embed):
        bz = input_ids.shape[0]
        mask_idx = torch.nonzero(input_ids == self.mask_id, as_tuple=False)
        mask_idx = mask_idx.reshape(bz, 2, -1)[:, :, 1]
        for b in range(bz):
            for i in range(2):
                input_embed[b, mask_idx[b, i], :] = self.mask[i, :]
        return input_embed

