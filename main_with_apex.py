"""
    所需pytorch>=1.6+
    需要注意混合精度训练时如果使用梯度裁剪，则代码可能不一样，可参考：
    https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training
"""
import os
import numpy as np
import logging
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
# 混合精度训练所需
from torch.cuda.amp import autocast as autocast, GradScaler

from config.config_with_apex import BertForClassificationConfig, BilstmForClassificationConfig
from processor.processor import PROCESSOR
from models.bertForClassification import BertForSequenceClassification
from models.bilstmForClassification import BilstmForSequenceClassification
from utils.utils import set_seed, set_logger

import warnings

warnings.filterwarnings("ignore")


# logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, criterion, device, config):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.device = device

    def load_ckp(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    def train(self, train_loader, dev_loader=None):
        # total_step = len(train_loader) * self.config.epochs
        if self.config.use_accumulation_steps:
            # 使用了梯度累加之后这里需要变换下
            total_step = len(train_loader) // self.config.accumulation_steps * self.config.epochs
        else:
            total_step = len(train_loader) * self.config.epochs
        self.optimizer, self.scheduler = self.build_optimizers(
            self.model,
            self.config,
            total_step
        )
        global_step = 0
        eval_step = 100
        best_dev_macro_f1 = 0.0
        # 在训练最开始之前实例化一个GradScaler对象
        scaler = GradScaler()
        for epoch in range(self.config.epochs):
            for train_step, train_data in enumerate(train_loader):
                self.model.train()
                if model_name == "bert_apex":
                    input_ids, token_type_ids, attention_mask, label_ids = train_data
                    input_ids = input_ids.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    label_ids = label_ids.to(self.device)
                    if self.config.use_apex:
                        if self.config.use_accumulation_steps:
                            with autocast():
                                train_outputs = self.model(input_ids, token_type_ids, attention_mask)
                                loss = self.criterion(train_outputs, label_ids)
                                loss = loss / self.config.accumulation_steps
                        else:
                            with autocast():
                                train_outputs = self.model(input_ids, token_type_ids, attention_mask)
                                loss = self.criterion(train_outputs, label_ids)
                    else:
                        if self.config.use_accumulation_steps:
                            train_outputs = self.model(input_ids, token_type_ids, attention_mask)
                            loss = self.criterion(train_outputs, label_ids)
                            loss = loss / self.config.accumulation_steps
                        else:
                            train_outputs = self.model(input_ids, token_type_ids, attention_mask)
                            loss = self.criterion(train_outputs, label_ids)
                elif model_name == "bilstm_apex":
                    input_ids, label_ids = train_data
                    sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
                    input_ids = input_ids.to(self.device)
                    label_ids = label_ids.to(self.device)
                    # 前向过程(model + loss)开启 autocast
                    if self.config.use_apex:
                        if self.config.use_accumulation_steps:
                            with autocast():
                                train_outputs = self.model(input_ids, sentence_lengths)
                                loss = self.criterion(train_outputs, label_ids)
                                loss = loss / self.config.accumulation_steps
                        else:
                            with autocast():
                                train_outputs = self.model(input_ids, sentence_lengths)
                                loss = self.criterion(train_outputs, label_ids)
                    else:
                        if self.config.use_accumulation_steps:
                            train_outputs = self.model(input_ids, sentence_lengths)
                            loss = self.criterion(train_outputs, label_ids)
                            loss = loss / self.config.accumulation_steps
                        else:
                            train_outputs = self.model(input_ids, sentence_lengths)
                            loss = self.criterion(train_outputs, label_ids)
                else:
                    raise Exception("请输入正确的模型名称")
                """
                    梯度累加主要的就是以下部分，需要注意：
                    1、梯度累加的步长，假设原始batch_size=32，梯度累加步长为4，
                    那么最终的batch_size就是32*4=128
                    2、随着batch_size的增大，学习率要相应放大，比如初始学习率为
                    2e-5，使用梯度累加后应该变为2e-5*4=8e-5
                    3、如果出现学习率过大导致不收敛问题，可使用warm_up，在使用
                    warm_up的时候，也要考虑到梯度累加的影响
                """
                if self.config.use_accumulation_steps:
                    if self.config.use_apex:
                        # Scales loss，这是因为半精度的数值范围有限，因此需要用它放大
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if (train_step + 1) % self.config.accumulation_steps == 0:
                        # self.optimizer.step()
                        # self.scheduler.step()
                        # scaler.step() unscale之前放大后的梯度，但是scale太多可能出现inf或NaN
                        # 故其会判断是否出现了inf/NaN
                        # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                        # 如果检测到出现了inf或者NaN，就跳过这次梯度更新，同时动态调整scaler的大小
                        if self.config.use_apex:
                            scaler.step(self.optimizer)
                            # 查看是否要更新scaler
                            scaler.update()
                        else:
                            self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        print(
                            "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step,
                                                                             loss.item()))
                        global_step += 1
                        if dev_loader:
                            if global_step % eval_step == 0:
                                dev_loss, dev_outputs, dev_targets = self.dev(dev_loader)
                                accuracy, precision, recall, macro_f1 = self.get_metrics(dev_outputs, dev_targets)
                                print(
                                    "【dev】 loss：{:.6f} accuracy：{:.4f} precision：{:.4f} recall：{:.4f} macro_f1：{:.4f}".format(
                                        dev_loss,
                                        accuracy,
                                        precision,
                                        recall,
                                        macro_f1))
                                if macro_f1 > best_dev_macro_f1:
                                    checkpoint = {
                                        'state_dict': self.model.state_dict(),
                                    }
                                    best_dev_macro_f1 = macro_f1
                                    checkpoint_path = os.path.join(self.config.save_dir,
                                                                   '{}_best.pt'.format(model_name))
                                    self.save_ckp(checkpoint, checkpoint_path)
                else:
                    if self.config.use_apex:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if self.config.use_apex:
                        scaler.step(self.optimizer)
                        # 查看是否要更新scaler
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    print(
                        "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss.item()))
                    global_step += 1
                    if dev_loader:
                        if global_step % eval_step == 0:
                            dev_loss, dev_outputs, dev_targets = self.dev(dev_loader)
                            accuracy, precision, recall, macro_f1 = self.get_metrics(dev_outputs, dev_targets)
                            print(
                                "【dev】 loss：{:.6f} accuracy：{:.4f} precision：{:.4f} recall：{:.4f} macro_f1：{:.4f}".format(
                                    dev_loss,
                                    accuracy,
                                    precision,
                                    recall,
                                    macro_f1))
                            if macro_f1 > best_dev_macro_f1:
                                checkpoint = {
                                    'state_dict': self.model.state_dict(),
                                }
                                best_dev_macro_f1 = macro_f1
                                checkpoint_path = os.path.join(self.config.save_dir, '{}_best.pt'.format(model_name))
                                self.save_ckp(checkpoint, checkpoint_path)

        checkpoint_path = os.path.join(self.config.save_dir, '{}_final.pt'.format(model_name))
        checkpoint = {
            'state_dict': self.model.state_dict(),
        }
        self.save_ckp(checkpoint, checkpoint_path)

    def dev(self, dev_loader):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(dev_loader):
                if model_name == "bert_apex":
                    input_ids, token_type_ids, attention_mask, label_ids = dev_data
                    input_ids = input_ids.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    label_ids = label_ids.to(self.device)
                    outputs = self.model(input_ids, token_type_ids, attention_mask)
                elif model_name == "bilstm_apex":
                    input_ids, label_ids = dev_data
                    sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
                    input_ids = input_ids.to(self.device)
                    label_ids = label_ids.to(self.device)
                    outputs = self.model(input_ids, sentence_lengths)
                else:
                    raise Exception("请输入正确的模型名称")
                loss = self.criterion(outputs, label_ids)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(label_ids.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path, test_loader):
        model = self.model
        model = self.load_ckp(model, checkpoint_path)
        model.eval()
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(test_loader):
                if model_name == "bert_apex":
                    input_ids, token_type_ids, attention_mask, label_ids = test_data
                    input_ids = input_ids.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    label_ids = label_ids.to(self.device)
                    outputs = model(input_ids, token_type_ids, attention_mask)
                elif model_name == "bilstm_apex":
                    input_ids, label_ids = test_data
                    sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
                    input_ids = input_ids.to(self.device)
                    label_ids = label_ids.to(self.device)
                    outputs = model(input_ids, sentence_lengths)
                else:
                    raise Exception("请输入正确的模型名称")
                loss = self.criterion(outputs, label_ids)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(label_ids.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text):
        model = self.model
        checkpoint = os.path.join(self.config.save_dir, '{}_best.pt'.format(model_name))
        model = self.load_ckp(model, checkpoint)
        model.eval()
        with torch.no_grad():
            if model_name == "bert_apex":
                inputs = tokenizer.encode_plus(text=text,
                                               max_length=self.config.max_seq_len,
                                               truncation='longest_first',
                                               padding="max_length",
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
                input_ids = inputs["input_ids"].to(self.device)
                token_type_ids = inputs["token_type_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                outputs = model(input_ids, token_type_ids, attention_mask)
            elif model_name == "bilstm_apex":
                inputs = [tokenizer["char2id"].get(char, 1) for char in text]
                if len(inputs) >= self.config.max_seq_len:
                    input_ids = inputs[:self.config.max_seq_len]
                else:
                    input_ids = inputs + [0] * (self.config.max_seq_len - len(inputs))
                input_ids = torch.tensor([input_ids])
                sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
                input_ids = input_ids.to(self.device)
                outputs = model(input_ids, sentence_lengths)
            else:
                raise Exception("请输入正确的模型名称")
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=-1).flatten().tolist()
            if len(outputs) != 0:
                outputs = [self.config.id2label[i] for i in outputs]
                return outputs
            else:
                return '不好意思，我没有识别出来'

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        precision = precision_score(targets, outputs, average='macro')
        recall = recall_score(targets, outputs, average='macro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, precision, recall, macro_f1

    def get_classification_report(self, outputs, targets, labels):
        report = classification_report(targets, outputs, target_names=labels, digits=4)
        return report

    def build_optimizers(self, model, config, t_total):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=config.lr,
                          eps=config.adam_epsilon)
        warmup_steps = int(config.warmup_proporation * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        return optimizer, scheduler


if __name__ == '__main__':
    # data_name = "bert"
    # model_name = 'bert_apex'

    data_name = "bilstm"
    model_name = 'bilstm_apex'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(123)
    # set_logger('{}.log'.format(model_name))

    if model_name == 'bert_apex':
        bertForClsConfig = BertForClassificationConfig()
        processor = PROCESSOR['BertProcessor']()
        max_seq_len = bertForClsConfig.max_seq_len
        tokenizer = BertTokenizer.from_pretrained(bertForClsConfig.model_dir)

        bert_config = BertConfig.from_pretrained(bertForClsConfig.model_dir)  # 预训练bert模型的config
        bert_config.num_labels = len(bertForClsConfig.labels)
        model = BertForSequenceClassification(bert_config)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, criterion, device, bertForClsConfig)

        if bertForClsConfig.do_train:
            train_examples = processor.read_data(os.path.join(bertForClsConfig.data_dir, 'cnews.train.txt'))
            train_dataset = processor.get_examples(
                train_examples,
                max_seq_len,
                tokenizer,
                './data/THUCNews/train_{}.pkl'.format(data_name),
                bertForClsConfig.label2id,
                'train')
            train_loader = DataLoader(train_dataset, batch_size=bertForClsConfig.train_batch_size, shuffle=True)

            if bertForClsConfig.do_eval:
                eval_examples = processor.read_data(os.path.join(bertForClsConfig.data_dir, 'cnews.val.txt'))
                eval_dataset = processor.get_examples(
                    eval_examples,
                    max_seq_len,
                    tokenizer,
                    './data/THUCNews/eval_{}.pkl'.format(data_name),
                    bertForClsConfig.label2id,
                    'eval')
                eval_loader = DataLoader(eval_dataset, batch_size=bertForClsConfig.eval_batch_size, shuffle=False)
                trainer.train(train_loader, eval_loader)
            else:
                trainer.train(train_loader)

        if bertForClsConfig.do_test:
            test_examples = processor.read_data(os.path.join(bertForClsConfig.data_dir, 'cnews.test.txt'))
            test_dataset = processor.get_examples(
                test_examples,
                max_seq_len,
                tokenizer,
                './data/THUCNews/test_{}.pkl'.format(data_name),
                bertForClsConfig.label2id,
                'test'
            )
            test_loader = DataLoader(test_dataset, batch_size=bertForClsConfig.eval_batch_size, shuffle=False)
            total_loss, test_outputs, test_targets = trainer.test(
                os.path.join(bertForClsConfig.save_dir, '{}_best.pt'.format(model_name)),
                test_loader,
            )
            _, _, _, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
            print('macro_f1：{}'.format(macro_f1))
            report = trainer.get_classification_report(test_outputs, test_targets, labels=bertForClsConfig.labels)
            print(report)

        if bertForClsConfig.do_predict:
            with open(os.path.join(bertForClsConfig.data_dir, 'cnews.test.txt'), 'r') as fp:
                lines = fp.readlines()
                ind = np.random.randint(0, len(lines))
                line = lines[ind].strip().split('\t')
                text = line[1]
                print(text)
                result = trainer.predict(tokenizer, text)
                print("预测标签：", result)
                print("真实标签：", line[0])
                print("==========================")

    elif model_name == 'bilstm_apex':
        bilstmForClsConfig = BilstmForClassificationConfig()
        processor = PROCESSOR['BilstmProcessor']()
        max_seq_len = bilstmForClsConfig.max_seq_len
        tokenizer = {}
        char2id = {}
        id2char = {}
        for i, char in enumerate(bilstmForClsConfig.vocab):
            char2id[char] = i
            id2char[i] = char
        tokenizer["char2id"] = char2id
        tokenizer["id2char"] = id2char

        model = BilstmForSequenceClassification(
            len(bilstmForClsConfig.labels),
            bilstmForClsConfig.vocab_size,
            bilstmForClsConfig.word_embedding_dimension,
            bilstmForClsConfig.hidden_dim
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, criterion, device, bilstmForClsConfig)

        if bilstmForClsConfig.do_train:
            train_examples = processor.read_data(os.path.join(bilstmForClsConfig.data_dir, 'cnews.train.txt'))
            train_dataset = processor.get_examples(
                train_examples,
                max_seq_len,
                tokenizer,
                './data/THUCNews/train_{}.pkl'.format(data_name),
                bilstmForClsConfig.label2id,
                'train')
            train_loader = DataLoader(train_dataset, batch_size=bilstmForClsConfig.train_batch_size, shuffle=True)

            if bilstmForClsConfig.do_eval:
                eval_examples = processor.read_data(os.path.join(bilstmForClsConfig.data_dir, 'cnews.val.txt'))
                eval_dataset = processor.get_examples(
                    eval_examples,
                    max_seq_len,
                    tokenizer,
                    './data/THUCNews/eval_{}.pkl'.format(data_name),
                    bilstmForClsConfig.label2id,
                    'eval')
                eval_loader = DataLoader(eval_dataset, batch_size=bilstmForClsConfig.eval_batch_size, shuffle=False)
                trainer.train(train_loader, eval_loader)
            else:
                trainer.train(train_loader)

        if bilstmForClsConfig.do_test:
            test_examples = processor.read_data(os.path.join(bilstmForClsConfig.data_dir, 'cnews.test.txt'))
            test_dataset = processor.get_examples(
                test_examples,
                max_seq_len,
                tokenizer,
                './data/THUCNews/test_{}.pkl'.format(data_name),
                bilstmForClsConfig.label2id,
                'test'
            )
            test_loader = DataLoader(test_dataset, batch_size=bilstmForClsConfig.eval_batch_size, shuffle=False)
            total_loss, test_outputs, test_targets = trainer.test(
                os.path.join(bilstmForClsConfig.save_dir, '{}_best.pt'.format(model_name)),
                test_loader,
            )
            _, _, _, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
            print('macro_f1：{}'.format(macro_f1))
            report = trainer.get_classification_report(test_outputs, test_targets, labels=bilstmForClsConfig.labels)
            print(report)

        if bilstmForClsConfig.do_predict:
            with open(os.path.join(bilstmForClsConfig.data_dir, 'cnews.test.txt'), 'r') as fp:
                lines = fp.readlines()
                ind = np.random.randint(0, len(lines))
                line = lines[ind].strip().split('\t')
                text = line[1]
                print(text)
                result = trainer.predict(tokenizer, text)
                print("预测标签：", result)
                print("真实标签：", line[0])
                print("==========================")
