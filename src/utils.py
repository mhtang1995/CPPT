import torch
import logging
import time
from datetime import timedelta



class LogFormatter(object):

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def set_logger(filepath):
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


def binarization(predict):
    max_index = torch.argmax(predict, dim=1)
    for dim, i in enumerate(max_index):
        predict[dim, i] = 1
    predict = predict.gt(0.5).int()
    return predict


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall)


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
            macro_p_list.append(t.item()/p.item())
    macro_p = sum(macro_p_list)/len(macro_p_list)
    # macro_p = ((label * predict).sum(1) / predict.sum(1)).mean()
    macro_r = ((label * predict).sum(1) / label.sum(1)).mean()
    macro_f1 = f1(macro_p, macro_r)

    logging.info('step %d\tmicro_f1: %f\tmacro_f1: %f\tstrict_acc: %f\tpos_acc: %f\tneg_acc: %f'
                 % (step, micro_f1, macro_f1, strict_acc, pos_acc, neg_acc))
    logging.info(f'step{step}: macro_P: {macro_p}, macro_R: {macro_r}, macro_F: {macro_f1}')
    return macro_p, macro_r, macro_f1, strict_acc
