import os
import numpy as np
import logging
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup

from config.config import BertForClassificationConfig, BilstmForClassificationConfig
# from processor.processor import PROCESSOR
from processor.kd_processor import PROCESSOR
from models.bertForClassification import BertForSequenceClassification
from models.bilstmForClassification import BilstmForSequenceClassification
from utils.utils import set_seed, set_logger

import warnings
warnings.filterwarnings("ignore")

class KDTrainer:
    def __init__(self,
                 teacher_model,
                 student_model,
                 criterion,
                 kd_criterion,
                 teacher_config,
                 student_config,
                 device):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.criterion = criterion
        self.kd_ceriterion = kd_criterion
        self.teacher_config = teacher_config
        self.student_config = student_config
        self.device = device
        self.T = 20

    def load_ckp(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    def train(self, train_loader, dev_loader=None):
        total_step = len(train_loader) * self.student_config.epochs
        self.optimizer, self.scheduler = self.build_optimizers(
            self.student_model,
            self.student_config,
            total_step
        )
        global_step = 0
        eval_step = 100
        best_dev_macro_f1 = 0.0
        for epoch in range(self.student_config.epochs):
            for train_step, train_data in enumerate(train_loader):
                teacher_input_ids, token_type_ids, attention_mask, student_input_ids, label_ids = train_data
                labels_ids = label_ids.to(self.device)
                # 不让教师模型训练
                self.teacher_model.eval()
                with torch.no_grad():
                    teacher_input_ids = teacher_input_ids.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    teacher_train_outputs = self.teacher_model(teacher_input_ids, token_type_ids, attention_mask)
                self.student_model.train()
                sentence_lengths = torch.sum((student_input_ids > 0).type(torch.long), dim=-1)
                student_input_ids = student_input_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                student_train_outputs = self.student_model(student_input_ids, sentence_lengths)
                hard_loss = self.criterion(student_train_outputs, label_ids)
                soft_loss = self.kd_ceriterion(F.log_softmax(student_train_outputs / self.T), F.softmax(teacher_train_outputs / self.T))
                loss = hard_loss + soft_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
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
                                'state_dict': self.student_model.state_dict(),
                            }
                            best_dev_macro_f1 = macro_f1
                            checkpoint_path = os.path.join(self.student_config.save_dir, '{}_best.pt'.format(model_name))
                            self.save_ckp(checkpoint, checkpoint_path)

        checkpoint_path = os.path.join(self.student_config.save_dir, '{}_final.pt'.format(model_name))
        checkpoint = {
            'state_dict': self.student_model.state_dict(),
        }
        self.save_ckp(checkpoint, checkpoint_path)

    def dev(self, dev_loader):
        self.student_model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(dev_loader):

                teacher_input_ids, token_type_ids, \
                attention_mask, student_input_ids, label_ids = dev_data
                sentence_lengths = torch.sum((student_input_ids > 0).type(torch.long), dim=-1)
                student_input_ids = student_input_ids.to(self.device)
                outputs = self.student_model(student_input_ids, sentence_lengths)
                label_ids = label_ids.to(self.device)
                loss = self.criterion(outputs, label_ids)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(label_ids.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path, test_loader):
        student_model = self.student_model
        student_model = self.load_ckp(student_model, checkpoint_path)
        student_model.eval()
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(test_loader):
                teacher_input_ids, token_type_ids, attention_mask, \
                student_input_ids, label_ids = test_data
                sentence_lengths = torch.sum((student_input_ids > 0).type(torch.long), dim=-1)
                student_input_ids = student_input_ids.to(self.device)
                label_ids =  label_ids.to(self.device)
                outputs = student_model(student_input_ids, sentence_lengths)

                loss = self.criterion(outputs, label_ids)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(label_ids.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text):
        student_model = self.student_model
        checkpoint = os.path.join(self.student_config.save_dir, '{}_best.pt'.format(model_name))
        student_model = self.load_ckp(student_model, checkpoint)
        student_model.eval()
        with torch.no_grad():
            inputs = [tokenizer["char2id"].get(char, 1) for char in text]
            if len(inputs) >= self.student_config.max_seq_len:
                input_ids = inputs[:self.student_config.max_seq_len]
            else:
                input_ids = inputs + [0] * (self.student_config.max_seq_len - len(inputs))
            input_ids = torch.tensor([input_ids])
            sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
            input_ids = input_ids.to(self.device)
            outputs = student_model(input_ids, sentence_lengths)
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=-1).flatten().tolist()
            if len(outputs) != 0:
                outputs = [self.student_config.id2label[i] for i in outputs]
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

def load_teacher_ckp(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_name = "kd"  # 主要是生成数据时保存的名字
    model_name = "kd_T_20"  # 主要是模型保存时保存的名字

    set_seed(123)
    # set_logger('{}.log'.format("kd_main"))

    teacherConfig = BertForClassificationConfig()
    studentConfig = BilstmForClassificationConfig()

    # 由于教师模型和学生模型的tokenizer是不同的
    processor = PROCESSOR['KDProcessor']()
    max_seq_len = studentConfig.max_seq_len
    teacher_tokenizer = BertTokenizer.from_pretrained(teacherConfig.model_dir)
    student_tokenizer = {}
    char2id = {}
    id2char = {}
    for i, char in enumerate(studentConfig.vocab):
        char2id[char] = i
        id2char[i] = char
    student_tokenizer["char2id"] = char2id
    student_tokenizer["id2char"] = id2char

    bert_config = BertConfig.from_pretrained(teacherConfig.model_dir)
    bert_config.num_labels = len(teacherConfig.labels)
    teacher_model = BertForSequenceClassification(bert_config)
    teacher_checkpoint_path = os.path.join(teacherConfig.save_dir, "bert_best.pt")
    teacher_model = load_teacher_ckp(teacher_model, teacher_checkpoint_path)
    teacher_model = teacher_model.to(device)
    student_model = BilstmForSequenceClassification(
        len(studentConfig.labels),
        studentConfig.vocab_size,
        studentConfig.word_embedding_dimension,
        studentConfig.hidden_dim
    )
    student_model = student_model.to(device)

    criterion = nn.CrossEntropyLoss()
    kd_criterion = nn.KLDivLoss()

    kdtrainer = KDTrainer(
        teacher_model,
        student_model,
        criterion,
        kd_criterion,
        teacherConfig,
        studentConfig,
        device,
    )

    if studentConfig.do_train:
        train_examples = processor.read_data(os.path.join(studentConfig.data_dir, 'cnews.train.txt'))
        train_dataset = processor.get_examples(
            train_examples,
            max_seq_len,
            teacher_tokenizer,
            student_tokenizer,
            './data/THUCNews/train_{}.pkl'.format(data_name),
            studentConfig.label2id,
            'train')
        train_loader = DataLoader(train_dataset, batch_size=studentConfig.train_batch_size, shuffle=True)

        if studentConfig.do_eval:
            eval_examples = processor.read_data(os.path.join(studentConfig.data_dir, 'cnews.val.txt'))
            eval_dataset = processor.get_examples(
                eval_examples,
                max_seq_len,
                teacher_tokenizer,
                student_tokenizer,
                './data/THUCNews/eval_{}.pkl'.format(data_name),
                studentConfig.label2id,
                'eval')
            eval_loader = DataLoader(eval_dataset, batch_size=studentConfig.eval_batch_size, shuffle=False)
            kdtrainer.train(train_loader, eval_loader)
        else:
            kdtrainer.train(train_loader)

    if studentConfig.do_test:
        test_examples = processor.read_data(os.path.join(studentConfig.data_dir, 'cnews.test.txt'))
        test_dataset = processor.get_examples(
            test_examples,
            max_seq_len,
            teacher_tokenizer,
            student_tokenizer,
            './data/THUCNews/test_{}.pkl'.format(data_name),
            studentConfig.label2id,
            'test'
        )
        test_loader = DataLoader(test_dataset, batch_size=studentConfig.eval_batch_size, shuffle=False)
        total_loss, test_outputs, test_targets = kdtrainer.test(
            os.path.join(studentConfig.save_dir, '{}_best.pt'.format(model_name)),
            test_loader,
        )
        _, _, _, macro_f1 = kdtrainer.get_metrics(test_outputs, test_targets)
        print('macro_f1：{}'.format(macro_f1))
        report = kdtrainer.get_classification_report(test_outputs, test_targets, labels=studentConfig.labels)
        print(report)

    if studentConfig.do_predict:
        with open(os.path.join(studentConfig.data_dir, 'cnews.test.txt'), 'r') as fp:
            lines = fp.readlines()
            ind = np.random.randint(0, len(lines))
            line = lines[ind].strip().split('\t')
            text = line[1]
            print(text)
            result = kdtrainer.predict(student_tokenizer, text)
            print("预测标签：", result)
            print("真实标签：", line[0])
            print("==========================")
