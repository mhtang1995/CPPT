import json


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.dataset = config["dataset"]
        self.mode = config['mode']
        self.times = config["times"]
        self.device = config["device"]

        self.mask_split = config["mask_split"]
        self.backbone = config["backbone"]
        self.bert_name = config["bert_name"]
        self.bert_size = config["bert_size"]
        self.threshold_one = config["threshold_one"]
        self.threshold_two = config["threshold_two"]
        self.weight = config["weight"]
        self.threshold = config["threshold"]

        self.ontology = config["ontology"]
        self.train = config["train"]
        self.valid = config["valid"]
        self.test = config["test"]
        self.fc_param = config["fc_param"]
        self.dense_param = config["dense_param"]
        self.ln_param = config['ln_param']

        self.lr = config["lr"]
        self.loss_version = config["loss_version"]

        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.max_epoch = config["max_epoch"]
        self.valid_step = config["valid_step"]
        self.log_step = config["log_step"]
        self.patience = config["patience"]

        self.save_dir = config["save_dir"]
        self.use_checkpoint = config["use_checkpoint"]

    def __repr__(self):
        return "{}".format(self.__dict__.items())
