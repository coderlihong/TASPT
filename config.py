from transformers import BertTokenizer
import  torch
class Config():
    def __init__(self):
        self.max_len = 512
        # self.bert_tokenizer = '/home/cslotus/pretrain_models/bert-base-uncased'
        # self.bert_model_path = '/home/cslotus/pretrain_models/bert-base-uncased'
        self.batch_size = 32
        self.bert_tokenizer = 'D:/code/pretrain_models/bert-ancient-chinese'
        self.bert_model_path = 'D:/code/pretrain_models/bert-ancient-chinese'
        self.hidden_size = 768
        self.epochs = 50
        self.lr = 3e-5
        self.cls_max_len = 512
        self.pre_seq_len = 48
        self.num_labels = 0
        self.prefix_hidden_size = 128
        # self.glm_config = 'D:/code/pretrain_models/chatgml3'
        # self.param_path = 'E:/finalshelldownload/params.pth'
        # self.llm_tokenizer = 'D:/code/pretrain_models/chatgml3'
        #
        self.glm_config = '/home/wh161054332/pretrain/chatgml3'
        self.param_path = '/home/wh161054332/project/dating/adver_train/params.pth'
        self.llm_tokenizer = '/home/wh161054332/pretrain/chatgml3'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.num_heads = 4
        self.adapter_size = 128