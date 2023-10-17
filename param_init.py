import torch
import os
from transformers import BertTokenizer, BertForMaskedLM

from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM


def init_decoder_weight(dataset, ontology_file, save_dir, MLM_decoder, bert_size):
    type_token = []
    with open(ontology_file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.replace('_', ' ')
            line = line.replace('/', ' ')
            # line = line.split('/')[-1]
            type_token.append(tokenizer(line, add_special_tokens=False)['input_ids'])
    type2token = torch.zeros(len(type_token), tokenizer.vocab_size)
    for idx, temp in enumerate(type_token):
        for i in temp:
            type2token[idx, i] = 1 / len(temp)
    fc = torch.nn.Linear(bert_size, len(type_token))
    MLM_decoder_weight = MLM_decoder.weight.data
    MLM_decoder_bias = MLM_decoder.bias.data
    fc.weight.data = type2token @ MLM_decoder_weight
    fc.bias.data = type2token @ MLM_decoder_bias
    torch.save(fc.state_dict(), os.path.join(save_dir, f'{dataset}_fc.pth'))


if __name__ == '__main__':
    save_dir = './save'
    # bert_name = 'bert-base-cased'
    bert_name = 'bert-base-cased'
    save_dir = os.path.join(save_dir, bert_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    bert_path = './Bert/bert-base-cased'
    bert_size = 768


    # init model params from MLMHead

    model = BertForMaskedLM.from_pretrained(bert_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # transform layer & LayerNorm
    dense = model.cls.predictions.transform.dense.state_dict()
    ln = model.cls.predictions.transform.LayerNorm.state_dict()

    MLM_decoder = model.get_output_embeddings()



    torch.save(dense, os.path.join(save_dir, 'dense.pth'))
    torch.save(ln, os.path.join(save_dir, 'ln.pth'))

    # decoder layer
    init_decoder_weight('ultra', './datasets/original/ontology/ultra_types.txt', save_dir, MLM_decoder, bert_size)
    init_decoder_weight('onto', './datasets/original/ontology/onto_ontology.txt', save_dir, MLM_decoder, bert_size)
    init_decoder_weight('bbn', './datasets/original/ontology/bbn_types.txt', save_dir, MLM_decoder, bert_size)
    init_decoder_weight('figer', './datasets/original/ontology/figer_types.txt', save_dir, MLM_decoder, bert_size)

