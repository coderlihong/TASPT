import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score

from dataset import combine_all,load_dataloader
from model import *
from config import Config
from torch.cuda.amp import GradScaler,autocast
from tqdm import tqdm
from torch import optim
from transformers import BertTokenizer
from torch.optim.lr_scheduler import StepLR
device = torch.device("cuda:0")
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        """
        Args:
            input (torch.Tensor): Model predictions (logits).
            target (torch.Tensor): Ground truth labels.
        Returns:
            torch.Tensor: Computed cross-entropy loss.
        """
        # Apply softmax to the input logits
        softmax_output = torch.softmax(input, dim=1)
        true = softmax_output.argmax(dim=-1)
        relative_distance = (true-target)**2
        scale_relative_distance = torch.tanh(relative_distance)
        scaler_ex = torch.exp(scale_relative_distance)


        # Compute the negative log likelihood loss
        loss = nn.NLLLoss(reduction='none')(torch.log(softmax_output), target)
        loss = scaler_ex*loss
        loss = loss.mean()
        return loss


def train(model,loader,config):
    opt = optim.AdamW(model.parameters(),lr=config.lr)
    cls_loss = CustomCrossEntropyLoss()
    distance_loss = nn.MSELoss()
    mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler()
    scheduler = StepLR(opt, step_size=10, gamma=0.5)

    for epoch in range(config.epochs):
        train_loss = 0
        train_acc = 0
        loader, label2idx = combine_all(config)
        model.train()
        for iter,batch in enumerate(tqdm(loader)):
            batch = [b.to(device) for b in batch]
            inputs, mlm_labels, attention_masks, relative_distance_label, labels = batch
            inputs = inputs.squeeze()
            mlm_labels = mlm_labels.squeeze()
            attention_masks = attention_masks.squeeze()
            opt.zero_grad()
            with autocast():
                mlm_output, relative_distance, label_cls = model(inputs,attention_masks)


            label_size = label_cls.size(-1)
            loss_cls = cls_loss(label_cls.view(-1,label_size),labels.view(-1))
            loss_distance = distance_loss(relative_distance_label.view(-1),relative_distance.view(-1))
            #
            loss_mlm = mlm_loss(mlm_output.view(-1,mlm_output.size(-1)),mlm_labels.view(-1))
            loss = loss_distance+loss_mlm
            scaler.scale(loss).backward()


            scaler.step(opt)
            scaler.update()
            train_loss+=loss.item()
        print(f'train_loss: {train_loss/len(loader)}')
        scheduler.step()
        torch.save(model.state_dict(),'model_en_wo_cls.pth')







def cls_train(model,train_loader,valid_loader):
    opt = optim.AdamW(model.parameters(),lr=config.lr,weight_decay=1e-4)
    cls_loss = CustomCrossEntropyLoss()
    bestf1 = 0
    scaler = GradScaler()
    scheduler = StepLR(opt, step_size=10, gamma=0.5)

    for epoch in range(config.epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for iter,batch in enumerate(tqdm(train_loader)):
            batch = [b.to(device) for b in batch]
            inputs, attention_masks, labels = batch
            opt.zero_grad()
            with autocast():
                logits= model(inputs,attention_masks)
            loss_cls = cls_loss(logits,labels)
            # loss = loss_distance
            loss = loss_cls
            scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            train_loss+=loss.item()
        scheduler.step()
        print(f'epoch: {epoch}, train_loss:{train_loss / len(train_loader)}')
        model.eval()
        valid_pred = []
        valid_true = []
        for batch in valid_loader:
            input_ids, attention_mask, label = [b.to(device) for b in batch]
            with torch.no_grad():
                with autocast():
                    cls_logits = model(input_ids, attention_mask)
            pred = cls_logits.argmax(dim=-1)
            valid_true.extend(label.detach().cpu().tolist())
            valid_pred.extend(pred.detach().cpu().tolist())
        p = precision_score(valid_true, valid_pred, average='macro')
        r = recall_score(valid_true, valid_pred, average='macro')
        f1 = f1_score(valid_true, valid_pred, average='macro')
        if bestf1 < f1:
            bestf1 = f1
            torch.save(model.state_dict(), 'model_en_ptv1mlp.pth')
        print(f'valid   precision:{p},recall:{r},f1:{f1}')


def cls_test(model,testloader,label2idx):
    model.load_state_dict(torch.load('model_en_ppt.pth'),strict=False)
    model.eval()
    valid_pred = []
    valid_true = []
    idx2label = {label2idx[item]:item for item in label2idx}
    for batch in testloader:
        input_ids, attention_mask, label = [b.to(device) for b in batch]
        with torch.no_grad():
            with autocast():
                cls_logits = model(input_ids, attention_mask)

        pred = cls_logits.argmax(dim=-1)
        valid_true.extend(label.detach().cpu().tolist())
        valid_pred.extend(pred.detach().cpu().tolist())
    y_true = []
    y_pred = []
    for i ,j in zip(valid_true,valid_pred):
        y_true.append(idx2label[i])
        y_pred.append(idx2label[j])
    pd.DataFrame({
        'true_label':y_true,
        'pred_label':y_pred
    }).to_csv('./results/model_en_ppt.csv',encoding='utf-8',index=False)
    print('写入完成')
    # valid_pred = [idx2label[item] for item in valid_pred]
    # df = pd.read_excel('../data/zh/trunc_test.xlsx')
    # df['pred'] = valid_pred
    # df.to_excel('res_zh_main.xlsx',index=False)
    p = precision_score(valid_true, valid_pred, average='macro')
    r = recall_score(valid_true, valid_pred, average='macro')
    f1 = f1_score(valid_true, valid_pred, average='macro')
    print(f'valid   precision:{p},recall:{r},f1:{f1}')


from sklearn.metrics.pairwise import cosine_similarity
def cls_test_word(model,testloader,label2idx,word,tokenizer):
    model.load_state_dict(torch.load('model_adv_promt.pth'),strict=False)

    word_ids = tokenizer.encode(word)[1]
    model.eval()
    valid_pred = []
    valid_true = []
    idx2label = {label2idx[item]:item for item in label2idx}
    mydict = {label2idx[item]:[] for item in label2idx}

    for batch in testloader:
        input_ids, attention_mask, label = [b.to(device) for b in batch]
        select_ids = torch.where(input_ids==word_ids,True,False)
        with torch.no_grad():
            with autocast():
                hidden_state,cls_logits = model.get_word_hidden(input_ids, attention_mask)
        for i,j,k in zip(hidden_state,select_ids,label):

            mydict[int(k)].extend(i[j].tolist())


            # print(hidden_state.shape)
        pred = cls_logits.argmax(dim=-1)
        valid_true.extend(label.detach().cpu().tolist())
        valid_pred.extend(pred.detach().cpu().tolist())
    new_dict = {}
    for i in mydict:
        new_dict[i]= np.mean(mydict[i],axis=0)

    base = new_dict[0]
    for i in new_dict:
        res = cosine_similarity(base.reshape(1, -1), new_dict[i].reshape(1, -1))
        print(i,res)
    # valid_pred = [idx2label[item] for item in valid_pred]
    # df = pd.read_excel('../data/zh/trunc_test.xlsx')
    # df['pred'] = valid_pred
    # df.to_excel('res_zh_main.xlsx',index=False)
    p = precision_score(valid_true, valid_pred, average='macro')
    r = recall_score(valid_true, valid_pred, average='macro')
    f1 = f1_score(valid_true, valid_pred, average='macro')
    print(f'valid   precision:{p},recall:{r},f1:{f1}')

def count_parameters(model, layers_to_count):
        """
        计算指定层的参数数量。

        参数:
        model (nn.Module): 要计算参数的模型。
        layers_to_count (list of str): 需要计算参数的层的名字列表。

        返回:
        int: 指定层的总参数数量。
        """
        all_params = sum(param.numel() for param in model.parameters())

        prompt_params = 0
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layers_to_count):
                prompt_params += param.numel()
        return prompt_params,all_params

if __name__ == '__main__':
    config = Config()

    loader,label2idx = combine_all(config)
    config.num_labels = len(label2idx)
    tokenizer = BertTokenizer.from_pretrained(config.bert_tokenizer)
    model = MyAdvPretrainModel(config,label2idx,tokenizer)
    # model.bert.encoder.load_state_dict(torch.load(f"{config.bert_model_path}/pytorch_model.bin"),strict=False)
    # model.bert.pooler.load_state_dict(torch.load(f"{config.bert_model_path}/pytorch_model.bin"), strict=False)
    # model.to(device)
    # model.load_state_dict(torch.load('model_en_wo_dis.pth'))
    #train(model,loader,config)
    train_loader, valid_loader, test_loader = load_dataloader(config,label2idx)
    #
    cls_model = MyPTV1LSTMmodelCLS(config,label2idx,tokenizer)
    #
    # cls_model = MyPPTmodel(config, label2idx, tokenizer)
    # cls_model = Mymodel(config, label2idx, tokenizer)


    # 定义需要计算参数量的层的名字列表
    layers_to_count = ['prefix_encoder','prefix_lstm']

    # 计算参数量
    prompt_params,all_params = count_parameters(cls_model, layers_to_count)

    print(f"prompt parameters in specified layers: {prompt_params}")
    print(f"total parameters in specified layers: {all_params}")
    print(prompt_params/all_params)
    # cls_model.adv_model.load_state_dict(torch.load('model_en_adver.pth'),strict=False)
    # cls_model.to(device)
    # print(cls_model)
    # cls_model.load_state_dict(torch.load('model_cls_en_wo_dis.pth'))
    # cls_train(cls_model,train_loader,valid_loader)
    # cls_test(cls_model,test_loader,label2idx)
    # word='曰'
    # cls_test_word(cls_model,test_loader,label2idx,word,tokenizer)






