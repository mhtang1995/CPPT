import random
import torch
import json
from json import JSONDecodeError
# import pyarrow as pa
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm


class UFET(Dataset):
    def __init__(self, file_path, type2id):
        super(UFET, self).__init__()
        self.type2id = type2id
        self.file_path = file_path

        self.data = []
        self.len = 0
        self.max_len = 0
        self.parse()

    def __len__(self):
        return self.len

    def parse(self):
        with open(self.file_path, 'r') as f:
            for line in f.readlines():
                try:
                    sample = json.loads(line.strip())
                except JSONDecodeError as e:
                    print("get a data error")
                    print(line)
                    continue
                if len(sample['y_str']) == 0:
                    print(sample)
                if len(sample['y_str']) > 0:
                    len_of_sample = len(sample['left_context_token']) + len(sample['right_context_token'])
                    if len_of_sample > self.max_len:
                        self.max_len = len_of_sample

                    left_context = " ".join(sample['left_context_token'])
                    mention = sample['mention_span']
                    right_context = " ".join(sample['right_context_token'])

                    self.data.append(
                        {
                            'left_context': left_context,
                            'mention': mention,
                            'right_context': right_context,
                            'y_str': sample['y_str']
                        }
                    )

        self.len = len(self.data)


    def __getitem__(self, idx):
        sample = self.data[idx]
        left_context = sample['left_context']
        mention = sample['mention']
        right_context = sample['right_context']

        sentence = f'{left_context} {mention} {right_context} [PROMPT] {mention} belongs to [PROMPT] [MASK]' \
                   f'rather than [PROMPT] [MASK]'


        label = torch.LongTensor([self.type2id[foo] for foo in sample['y_str'] if foo in self.type2id])
        label = torch.zeros(len(self.type2id)).scatter_(0, label, 1)
        return idx, sentence, label



    @staticmethod
    def collate_fn(train_data, tokenizer):
        idx = torch.LongTensor([data[0] for data in train_data])
        sentences = [data[1] for data in train_data]
        samples = tokenizer(sentences, truncation=True, padding=True, return_tensors='pt')
        input_ids, attention_mask = samples['input_ids'], samples['attention_mask']
        labels = torch.stack([data[2] for data in train_data], dim=0)
        return idx, input_ids, attention_mask, labels
