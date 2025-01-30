from transformers import BertModel, BertLMHeadModel, BertTokenizer, BertConfig, AutoTokenizer, AutoModel,RobertaModel
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Any, Optional, Tuple
from glm3 import ChatGLMModel
from torch.cuda.amp import GradScaler, autocast
from PEFTModel import *

class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class MyAdvPretrainModel(nn.Module):
    def __init__(self, config, label2idx, tokenizer, mode='train'):
        super().__init__()
        self.config = config
        bert_config = BertConfig.from_pretrained(config.bert_tokenizer)
        self.bert = BertModel.from_pretrained(config.bert_tokenizer)
        self.relative_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.relative_sequential = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, 1)
        )

        self.label_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.label_cls = nn.Sequential(nn.ReLU(),
                                       nn.LayerNorm(config.hidden_size),
                                       nn.Linear(config.hidden_size, len(label2idx)))

        self.mlm_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_cls = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, len(tokenizer))
        )
        self.grl = GRL()
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.mode = mode
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):

        batch, seq_len = input_ids.size()
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        if self.mode == 'train':
            output = self.bert(input_ids, attention_mask)
            hidden_states = output.last_hidden_state
            hidden_prompt = self.pooler(hidden_states)
            cls_pooler = hidden_prompt[:, 0, :]

            sep_hidden = hidden_prompt[:, int(self.config.max_len / 2)]

            mlm_prompt = self.mlm_transform(hidden_states)
            mlm_output = self.mlm_cls(mlm_prompt)

            cls_sep = torch.stack([cls_pooler, sep_hidden], dim=1)  # batch,2,768

            relative_prompt = self.relative_transform(cls_pooler - sep_hidden)
            relative_distance = self.relative_sequential(relative_prompt).squeeze()  #

            cls_sep = self.grl(cls_sep)
            label_cls = self.label_cls(cls_sep)

            return mlm_output, relative_distance, label_cls
        else:
            output = self.bert(input_ids, attention_mask)
            hidden_states = output.last_hidden_state
            hidden_prompt = self.pooler(hidden_states)
            cls_pooler = hidden_prompt[:, 0, :]
            sep_hidden = hidden_prompt[:, int(self.config.max_len / 2)]
            relative_prompt = self.relative_transform(cls_pooler - sep_hidden).unsqueeze(dim=1)  # batch,1,hidden
            mlm_prompt = self.mlm_transform(hidden_states)
            prompts = (hidden_prompt, relative_prompt, mlm_prompt)
            return prompts
    def test_vector(self,input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        hidden_states = output.last_hidden_state
        hidden_prompt = self.pooler(hidden_states)
        hidden_states=hidden_prompt[:, 0, :]
        return hidden_states

    def get_word_hidden(self,input_ids,attention_mask):
        output = self.bert(input_ids, attention_mask)
        hidden_states = self.mlm_transform(output.last_hidden_state)
        return hidden_states
    def test_attention(self,input_ids,attention_mask):
        output = self.bert(input_ids, attention_mask,output_attentions=True)
        attentions = output.attentions[-1].mean(dim=1)[:,0]


        return attentions[:,1:-1]

    def get_word_embedding(self,input_ids,attention_mask):
        output = self.bert(input_ids, attention_mask)
        hidden_states = output.last_hidden_state
        return hidden_states

class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, bertconfig, config):
        super().__init__()

        # Use a two-layer MLP to encode the prefix
        kv_size = bertconfig.num_hidden_layers * 2 * bertconfig.hidden_size
        self.embedding = torch.nn.Embedding(config.pre_seq_len, config.prefix_hidden_size)
        self.vector_dense = nn.Linear(bertconfig.hidden_size, config.prefix_hidden_size)
        self.attention = nn.MultiheadAttention(config.prefix_hidden_size, config.num_heads, batch_first=True,
                                               dropout=0.1)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(config.prefix_hidden_size, config.hidden_size), torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_size, kv_size))

    def forward(self, prefix: torch.Tensor, time_vector):
        prefix_tokens = self.embedding(prefix)
        time_vector = self.vector_dense(time_vector)
        prefix_tokens, _ = self.attention(prefix_tokens, time_vector, time_vector)
        past_key_values = self.trans(prefix_tokens)

        return past_key_values


class PrefixPPTEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, bertconfig, config):
        super().__init__()

        # Use a two-layer MLP to encode the prefix
        kv_size = bertconfig.num_hidden_layers * 2 * bertconfig.hidden_size
        self.embedding = torch.nn.Embedding(config.pre_seq_len, bertconfig.hidden_size)
        # self.vector_dense = nn.Linear(bertconfig.hidden_size, config.prefix_hidden_size)
        #
        # self.trans = torch.nn.Sequential(
        #     torch.nn.Linear(config.prefix_hidden_size, kv_size))


    def forward(self, prefix: torch.Tensor, time_vector):
        prefix_tokens = self.embedding(prefix)
        output = prefix_tokens+time_vector


        return output

class MyTuningmodel(nn.Module):
    def __init__(self, config, bertconfig):
        super().__init__()
        self.config = config
        self.bertconfig = bertconfig

        self.bert = BertModel(bertconfig)

        tokenizer = BertTokenizer.from_pretrained(config.bert_tokenizer)
        self.dense_invariance = nn.Linear(config.cls_max_len, int(config.pre_seq_len / 2))
        self.dense_variance = nn.Linear(1, int(config.pre_seq_len / 2))

        self.dropout = nn.Dropout(0.1)
        self.dropout_prompt = nn.Dropout(0.1)
        self.dropout_invariance = nn.Dropout(0.1)
        self.dropout_variance = nn.Dropout(0.1)

        for param in self.bert.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
        self.pre_seq_len = config.pre_seq_len
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(bertconfig, config)
        self.n_layer = bertconfig.num_hidden_layers
        self.n_head = bertconfig.num_attention_heads
        self.n_embd = bertconfig.hidden_size // bertconfig.num_attention_heads
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def get_prompt(self, batch_size, time_vector):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size,
                                                               -1).to(self.config.device)
        past_key_values = self.prefix_encoder(prefix_tokens, time_vector)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout_prompt(past_key_values)

        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, x, mask, hidden_states):
        hidden_prompt, relative_prompt, mlm_prompt = hidden_states
        invariance = mlm_prompt + hidden_prompt
        variance = relative_prompt

        invariance_prompt = self.dropout_invariance(self.dense_invariance(invariance.permute(0, 2, 1))).permute(0, 2, 1)
        variance_prompt = self.dropout_variance(self.dense_variance(variance.permute(0, 2, 1))).permute(0, 2, 1)
        vector = torch.cat([invariance_prompt, variance_prompt], dim=1)

        batch_size = x.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size, time_vector=vector)

        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)

        attention_mask = torch.cat((prefix_attention_mask, mask), dim=1)[:, :self.config.max_len]

        token_type_ids = torch.ones(batch_size, self.config.max_len - self.pre_seq_len, dtype=torch.int64).to(
            self.bert.device)
        x = x[:, :self.config.max_len - self.pre_seq_len]
        outputs = self.bert(
            input_ids=x,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class MyPTV1MLPmodel(nn.Module):
    def __init__(self, config, bertconfig):
        super().__init__()
        self.config = config
        self.bertconfig = bertconfig

        self.bert = BertModel(bertconfig)

        tokenizer = BertTokenizer.from_pretrained(config.bert_tokenizer)
        self.dense_invariance = nn.Linear(config.cls_max_len, int(config.pre_seq_len / 2))
        self.dense_variance = nn.Linear(1, int(config.pre_seq_len / 2))

        self.dropout = nn.Dropout(0.1)
        self.dropout_prompt = nn.Dropout(0.1)
        self.dropout_invariance = nn.Dropout(0.1)
        self.dropout_variance = nn.Dropout(0.1)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        self.prefix_mlp = nn.Sequential(
            nn.Linear(bertconfig.hidden_size,bertconfig.hidden_size),
            nn.ReLU()
        )
        self.n_layer = bertconfig.num_hidden_layers
        self.n_head = bertconfig.num_attention_heads
        self.n_embd = bertconfig.hidden_size // bertconfig.num_attention_heads
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)



    def forward(self, x, mask, hidden_states):
        hidden_prompt, relative_prompt, mlm_prompt = hidden_states
        invariance = mlm_prompt + hidden_prompt
        variance = relative_prompt

        invariance_prompt = self.dropout_invariance(self.dense_invariance(invariance.permute(0, 2, 1))).permute(0, 2, 1)
        variance_prompt = self.dropout_variance(self.dense_variance(variance.permute(0, 2, 1))).permute(0, 2, 1)
        vector = torch.cat([invariance_prompt, variance_prompt], dim=1)
        vector = self.prefix_mlp(vector)

        batch_size = x.shape[0]

        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)

        attention_mask = torch.cat((prefix_attention_mask, mask), dim=1)[:, :self.config.max_len]

        token_type_ids = torch.ones(batch_size, self.config.max_len - self.pre_seq_len, dtype=torch.int64).to(
            self.bert.device)
        x = x[:, :self.config.max_len - self.pre_seq_len]
        embedding = self.bert.embeddings(x,token_type_ids=token_type_ids,)
        embedding = torch.cat([vector,embedding],dim=1)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float16).min
        outputs = self.bert.encoder(
            hidden_states=embedding,

            attention_mask=extended_attention_mask
        )
        outputs = self.bert.pooler(outputs[0])
        pooled_output = outputs

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits



class MyPTV1SLTMmodel(nn.Module):
    def __init__(self, config, bertconfig):
        super().__init__()
        self.config = config
        self.bertconfig = bertconfig

        self.bert = BertModel(bertconfig)

        tokenizer = BertTokenizer.from_pretrained(config.bert_tokenizer)
        self.dense_invariance = nn.Linear(config.cls_max_len, int(config.pre_seq_len / 2))
        self.dense_variance = nn.Linear(1, int(config.pre_seq_len / 2))

        self.dropout = nn.Dropout(0.1)
        self.dropout_prompt = nn.Dropout(0.1)
        self.dropout_invariance = nn.Dropout(0.1)
        self.dropout_variance = nn.Dropout(0.1)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        self.prefix_lstm = nn.LSTM(input_size=bertconfig.hidden_size,hidden_size=bertconfig.hidden_size,num_layers=2,batch_first=True)
        self.n_layer = bertconfig.num_hidden_layers
        self.n_head = bertconfig.num_attention_heads
        self.n_embd = bertconfig.hidden_size // bertconfig.num_attention_heads
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)



    def forward(self, x, mask, hidden_states):
        hidden_prompt, relative_prompt, mlm_prompt = hidden_states
        invariance = mlm_prompt + hidden_prompt
        variance = relative_prompt

        invariance_prompt = self.dropout_invariance(self.dense_invariance(invariance.permute(0, 2, 1))).permute(0, 2, 1)
        variance_prompt = self.dropout_variance(self.dense_variance(variance.permute(0, 2, 1))).permute(0, 2, 1)
        vector = torch.cat([invariance_prompt, variance_prompt], dim=1)
        vector,_ = self.prefix_lstm(vector)

        batch_size = x.shape[0]

        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)

        attention_mask = torch.cat((prefix_attention_mask, mask), dim=1)[:, :self.config.max_len]

        token_type_ids = torch.ones(batch_size, self.config.max_len - self.pre_seq_len, dtype=torch.int64).to(
            self.bert.device)
        x = x[:, :self.config.max_len - self.pre_seq_len]
        embedding = self.bert.embeddings(x,token_type_ids=token_type_ids,)
        embedding = torch.cat([vector,embedding],dim=1)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float16).min
        outputs = self.bert.encoder(
            hidden_states=embedding,

            attention_mask=extended_attention_mask
        )
        outputs = self.bert.pooler(outputs[0])
        pooled_output = outputs

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
class MyPPTTuningmodel(nn.Module):
    def __init__(self, config, bertconfig):
        super().__init__()
        self.config = config
        self.bertconfig = bertconfig

        self.bert = BertModel(bertconfig)

        tokenizer = BertTokenizer.from_pretrained(config.bert_tokenizer)
        self.dense_invariance = nn.Linear(config.cls_max_len, int(config.pre_seq_len / 2))
        self.dense_variance = nn.Linear(1, int(config.pre_seq_len / 2))

        self.dropout = nn.Dropout(0.1)
        self.dropout_prompt = nn.Dropout(0.1)
        self.dropout_invariance = nn.Dropout(0.1)
        self.dropout_variance = nn.Dropout(0.1)

        for param in self.bert.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
        self.pre_seq_len = config.pre_seq_len
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixPPTEncoder(bertconfig, config)
        self.n_layer = bertconfig.num_hidden_layers
        self.n_head = bertconfig.num_attention_heads
        self.n_embd = bertconfig.hidden_size // bertconfig.num_attention_heads
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def get_prompt(self, batch_size, time_vector):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size,
                                                               -1).to(self.config.device)
        output = self.prefix_encoder(prefix_tokens, time_vector)

        return output

    def forward(self, x, mask, hidden_states):
        hidden_prompt, relative_prompt, mlm_prompt = hidden_states
        invariance = mlm_prompt + hidden_prompt
        variance = relative_prompt

        invariance_prompt = self.dropout_invariance(self.dense_invariance(invariance.permute(0, 2, 1))).permute(0, 2, 1)
        variance_prompt = self.dropout_variance(self.dense_variance(variance.permute(0, 2, 1))).permute(0, 2, 1)
        vector = torch.cat([invariance_prompt, variance_prompt], dim=1)

        batch_size = x.shape[0]
        output = self.get_prompt(batch_size=batch_size, time_vector=vector)

        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)

        attention_mask = torch.cat((prefix_attention_mask, mask), dim=1)[:, :self.config.max_len]

        token_type_ids = torch.ones(batch_size, self.config.max_len - self.pre_seq_len, dtype=torch.int64).to(
            self.bert.device)
        x = x[:, :self.config.max_len - self.pre_seq_len]
        embedding = self.bert.embeddings(x,token_type_ids=token_type_ids,)
        embedding = torch.cat([output,embedding],dim=1)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float16).min
        outputs = self.bert.encoder(
            hidden_states=embedding,

            attention_mask=extended_attention_mask
        )
        outputs = self.bert.pooler(outputs[0])
        pooled_output = outputs

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits



class Mymodel(nn.Module):
    def __init__(self, config, label2idx, tokenzier):
        super().__init__()
        self.adv_model = MyAdvPretrainModel(config, label2idx, tokenzier, 'generate')
        # for param in self.adv_model.parameters():
        #     param.requires_grad=False

        self.bertconfig = BertConfig.from_pretrained(config.bert_tokenizer)
        self.bertconfig.prefix_hidden_size = config.prefix_hidden_size
        self.tuning_model = MyTuningmodel(config, self.bertconfig)

    def forward(self, x, attention_masks):
        hidden_states = self.adv_model(x, attention_masks)

        logits = self.tuning_model(x, attention_masks, hidden_states)
        return logits

    def get_attention_score(self,x,attention_masks):
        scores = self.adv_model.test_attention(x,attention_masks)
        return scores

    def get_word_hidden(self,x,attention_masks):
        hidden_states = self.adv_model(x, attention_masks)
        logits = self.tuning_model(x, attention_masks, hidden_states)
        return self.adv_model.get_word_hidden(x,attention_masks),logits
class MyPPTmodel(nn.Module):
    def __init__(self, config, label2idx, tokenzier):
        super().__init__()
        self.adv_model = MyAdvPretrainModel(config, label2idx, tokenzier, 'generate')
        # for param in self.adv_model.parameters():
        #     param.requires_grad=False

        self.bertconfig = BertConfig.from_pretrained(config.bert_tokenizer)
        self.bertconfig.prefix_hidden_size = config.prefix_hidden_size
        self.tuning_model = MyPPTTuningmodel(config, self.bertconfig)

    def forward(self, x, attention_masks):
        hidden_states = self.adv_model(x, attention_masks)

        logits = self.tuning_model(x, attention_masks, hidden_states)
        return logits

    def get_attention_score(self,x,attention_masks):
        scores = self.adv_model.test_attention(x,attention_masks)
        return scores

    def get_word_hidden(self,x,attention_masks):
        hidden_states = self.adv_model(x, attention_masks)
        logits = self.tuning_model(x, attention_masks, hidden_states)
        return self.adv_model.get_word_hidden(x,attention_masks),logits

class MyPTV1MLPmodelCLS(nn.Module):
    def __init__(self, config, label2idx, tokenzier):
        super().__init__()
        self.adv_model = MyAdvPretrainModel(config, label2idx, tokenzier, 'generate')
        # for param in self.adv_model.parameters():
        #     param.requires_grad=False

        self.bertconfig = BertConfig.from_pretrained(config.bert_tokenizer)
        self.bertconfig.prefix_hidden_size = config.prefix_hidden_size
        self.tuning_model = MyPTV1MLPmodel(config, self.bertconfig)

    def forward(self, x, attention_masks):
        hidden_states = self.adv_model(x, attention_masks)

        logits = self.tuning_model(x, attention_masks, hidden_states)
        return logits

    def get_attention_score(self,x,attention_masks):
        scores = self.adv_model.test_attention(x,attention_masks)
        return scores

    def get_word_hidden(self,x,attention_masks):
        hidden_states = self.adv_model(x, attention_masks)
        logits = self.tuning_model(x, attention_masks, hidden_states)
        return self.adv_model.get_word_hidden(x,attention_masks),logits


class MyPTV1LSTMmodelCLS(nn.Module):
    def __init__(self, config, label2idx, tokenzier):
        super().__init__()
        self.adv_model = MyAdvPretrainModel(config, label2idx, tokenzier, 'generate')
        # for param in self.adv_model.parameters():
        #     param.requires_grad=False

        self.bertconfig = BertConfig.from_pretrained(config.bert_tokenizer)
        self.bertconfig.prefix_hidden_size = config.prefix_hidden_size
        self.tuning_model = MyPTV1SLTMmodel(config, self.bertconfig)

    def forward(self, x, attention_masks):
        hidden_states = self.adv_model(x, attention_masks)

        logits = self.tuning_model(x, attention_masks, hidden_states)
        return logits

    def get_attention_score(self,x,attention_masks):
        scores = self.adv_model.test_attention(x,attention_masks)
        return scores

    def get_word_hidden(self,x,attention_masks):
        hidden_states = self.adv_model(x, attention_masks)
        logits = self.tuning_model(x, attention_masks, hidden_states)
        return self.adv_model.get_word_hidden(x,attention_masks),logits
class MyBitfitmodel(nn.Module):
    def __init__(self, config, label2idx, tokenzier):
        super().__init__()
        self.adv_model = MyAdvPretrainModel(config, label2idx, tokenzier, 'generate')
        # for param in self.adv_model.parameters():
        #     param.requires_grad=False

        self.bertconfig = BertConfig.from_pretrained(config.bert_tokenizer)
        self.bertconfig.prefix_hidden_size = config.prefix_hidden_size
        self.tuning_model = MyTuningmodel(config, self.bertconfig)

        for name, param in self.named_parameters():
            if 'bias' in name:
                param.requires_grad = True

    def forward(self, x, attention_masks):
        hidden_states = self.adv_model(x, attention_masks)

        logits = self.tuning_model(x, attention_masks, hidden_states)
        return logits
class MyParallModel(nn.Module):
    def __init__(self, config, label2idx, tokenzier):
        super().__init__()
        self.adv_model = MyAdvPretrainModel(config, label2idx, tokenzier, 'generate')
        # for param in self.adv_model.parameters():
        #     param.requires_grad=False

        self.bertconfig = BertConfig.from_pretrained(config.bert_tokenizer)
        self.bertconfig.adapter_size = config.adapter_size

        self.tuning_model = ParallerModel(self.bertconfig)
        self.fc = nn.Linear(config.hidden_size,len(label2idx))
    def forward(self, x, attention_masks):
        hidden_states = self.adv_model(x, attention_masks)
        hidden_prompt, relative_prompt, mlm_prompt = hidden_states
        invariance = mlm_prompt + hidden_prompt
        variance = relative_prompt
        hidden_states = invariance+variance
        logits = self.tuning_model(x, attention_masks, temporal_hidden=hidden_states)
        output = self.fc(logits.pooler_output)
        return output


class MyLoraModel(nn.Module):
    def __init__(self, config, label2idx, tokenzier):
        super().__init__()
        self.adv_model = MyAdvPretrainModel(config, label2idx, tokenzier, 'generate')
        # for param in self.adv_model.parameters():
        #     param.requires_grad=False

        self.bertconfig = BertConfig.from_pretrained(config.bert_tokenizer)
        self.bertconfig.adapter_size = config.adapter_size

        self.tuning_model = LoraModel(self.bertconfig)
        self.fc = nn.Linear(config.hidden_size,len(label2idx))
    def forward(self, x, attention_masks):
        hidden_states = self.adv_model(x, attention_masks)
        hidden_prompt, relative_prompt, mlm_prompt = hidden_states
        invariance = mlm_prompt + hidden_prompt
        variance = relative_prompt
        hidden_states = invariance+variance
        logits = self.tuning_model(x, attention_masks, temporal_hidden=hidden_states)
        output = self.fc(logits.pooler_output)
        return output
class MyLSTM(nn.Module):
    def __init__(self, config, label2idx, tokenzier):
        super().__init__()
        self.embedding = nn.Embedding(len(tokenzier),config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size,256,num_layers=2,batch_first=True,bidirectional=True)
        self.ln = nn.Linear(512,len(label2idx))

    def forward(self, x):
        emebdding = self.embedding(x)
        h,_ = self.lstm(emebdding)
        output = self.ln(h[:,-1,:])


        return output

class MymodelLLM(nn.Module):
    def __init__(self, config, label2idx, tokenzier):
        super().__init__()
        self.adv_model = MyAdvPretrainModel(config, label2idx, tokenzier, 'generate')
        # for param in self.adv_model.parameters():
        #     param.requires_grad=False
        self.config = config

        # self.glmconfig = ChatGLM2Config.from_pretrained(config.glm_config)
        self.glmconfig.temporal_hidden = 768
        self.glmconfig.n_layer = self.glmconfig.num_layers
        self.glmconfig.prefix_projection = True

        self.prefix_encoder = PrefixEncoder(self.glmconfig)

        glm_model = ChatGLMModel(self.glmconfig).cuda()
        state = torch.load(config.param_path)

        for name in list(state.keys()):
            if 'transformer.encoder' in name:
                tmpstr = name.replace("transformer.encoder.", 'encoder.')
                state.update({tmpstr: state.pop(name)})
            elif 'transformer' in name:
                tmpstr = name.replace("transformer.", '')
                state.update({tmpstr: state.pop(name)})
        glm_model.load_state_dict(state, strict=False)

        self.bertconfig = BertConfig.from_pretrained(config.bert_tokenizer)
        self.bertconfig.prefix_hidden_size = config.prefix_hidden_size
        self.dense_layer = nn.Linear(config.max_len, config.pre_seq_len)
        self.tuning_model = glm_model
        self.dropout = nn.Dropout(0.1)
        llm_tokenizer = AutoTokenizer.from_pretrained(config.llm_tokenizer, trust_remote_code=True)
        self.cls = nn.Linear(4096, len(llm_tokenizer))

    def get_prompt(self, batch_size, time_vector):
        past_key_values = self.prefix_encoder(time_vector)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(batch_size, self.config.pre_seq_len,
                                               self.glmconfig.num_layers * 2,
                                               self.glmconfig.multi_query_group_num,
                                               self.glmconfig.kv_channels)
        past_key_values = self.dropout(past_key_values)

        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, bert_x, bert_mask, llm_x, llm_attention_mask):
        hidden_states = self.adv_model(bert_x, bert_mask)
        hidden_prompt, relative_prompt, mlm_prompt = hidden_states

        temporal_hidden = hidden_prompt + relative_prompt + mlm_prompt
        temporal_hidden = self.dense_layer(temporal_hidden.transpose(1, 2)).transpose(1, 2)
        batch_size = temporal_hidden.size(0)
        temporal_hidden = self.get_prompt(batch_size, temporal_hidden)
        logits = self.tuning_model(llm_x, attention_mask=llm_attention_mask, temporal_vector=temporal_hidden)

        cls = self.cls(logits[0].transpose(0, 1)[:, :16])

        return cls


class MyBert(nn.Module):
    def __init__(self, config, label2idx, tokenzier):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.bert_model_path)
        self.ln = nn.Linear(768,len(label2idx))

    def forward(self, x,attention_mask):
        output = self.bert(x,attention_mask)
        pooler = output.pooler_output
        output = self.ln(pooler)


        return output

from config import *


def get_label2idx():
    f = open('../data/zh/label.txt', encoding='utf-8').readlines()
    f = {i.strip(): iter for iter, i in enumerate(f)}
    return f


if __name__ == '__main__':
    config = Config()
    label2idx = get_label2idx()
    tokenizer = BertTokenizer.from_pretrained(config.bert_tokenizer)
    model = MymodelLLM(config, label2idx, tokenizer).cuda()
    input_ids = torch.randint(1423, 2000, (1, 512)).cuda()
    attention_mask = torch.ones((1, 512)).cuda()
    with autocast():
        output = model(input_ids, attention_mask, input_ids, attention_mask)
    print(output.shape)
    print(output)
