import numpy as np
import os
import pickle
import torch
from torch.utils.data import TensorDataset


def convert_array_to_tensor(inputs, dtype=None):
    if not inputs is np.ndarray:
        inputs = np.array(inputs)
    if dtype:
        inputs = torch.tensor(inputs, dtype=dtype)
    else:
        inputs = torch.tensor(inputs)
    return inputs


class KDProcessor:
    def read_data(self, file_path):
        with open(file_path, 'r') as fp:
            raw_examples = fp.read().strip()
        return raw_examples

    def get_examples(self,
                     raw_examples,
                     max_seq_len,
                     teacher_tokenizer,
                     student_tokenizer,
                     pickle_path,
                     label2id,
                     set_type,
                     sep='\t',
                     reverse=False):
        if not os.path.exists(pickle_path):
            total = len(raw_examples.split('\n'))

            teacher_input_ids_all = []
            student_input_ids_all = []
            token_type_ids_all = []
            attention_mask_all = []
            label_ids_all = []
            for i, line in enumerate(raw_examples.split('\n')):
                print("process:{}/{}".format(i, total))
                line = line.split(sep)
                if reverse:
                    text = line[0]
                    label = line[1]
                else:
                    label = line[0]
                    text = line[1]
                teacher_input_ids, token_type_ids, attention_mask = \
                    self.convert_text_to_teacher_feature(text, max_seq_len, teacher_tokenizer)
                student_input_ids = self.convert_text_to_student_feature(
                    text, max_seq_len, student_tokenizer
                )
                if i < 3:
                    print(f"{set_type} example-{i}")
                    print(f"text：{text[:max_seq_len]}")
                    print(f"teacher_input_ids:{teacher_input_ids}")
                    print(f"token_type_ids：{token_type_ids}")
                    print(f"attention_mask：{attention_mask}")
                    print(f"student_input_ids:{student_input_ids}")
                    print(f"label：{label}")
                teacher_input_ids_all.append(teacher_input_ids)
                student_input_ids_all.append(student_input_ids)
                token_type_ids_all.append(token_type_ids)
                attention_mask_all.append(attention_mask)
                label_ids_all.append(label2id[label])
            tensorDataset = TensorDataset(
                convert_array_to_tensor(teacher_input_ids_all),
                convert_array_to_tensor(token_type_ids_all),
                convert_array_to_tensor(attention_mask_all, dtype=torch.uint8),
                convert_array_to_tensor(student_input_ids_all),
                convert_array_to_tensor(label_ids_all),
            )
            with open(pickle_path, 'wb') as fp:
                pickle.dump(tensorDataset, fp)
        else:
            with open(pickle_path, 'rb') as fp:
                tensorDataset = pickle.load(fp)
        return tensorDataset

    def convert_text_to_teacher_feature(self, text, max_seq_len, tokenizer):
        encode_dict = tokenizer.encode_plus(text=text,
                                            max_length=max_seq_len,
                                            truncation='longest_first',
                                            padding="max_length",
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        # input_ids = convert_array_to_tensor(encode_dict['input_ids'], dtype=torch.int64)
        # token_type_ids = convert_array_to_tensor(encode_dict['token_type_ids'],dtype=torch.int64)
        # attention_mask = convert_array_to_tensor(encode_dict['attention_mask'], dtype=torch.uint8)

        input_ids = np.array(encode_dict['input_ids'])
        token_type_ids = np.array(encode_dict['token_type_ids'])
        attention_mask = np.array(encode_dict['attention_mask'])

        assert len(input_ids) == max_seq_len
        assert len(token_type_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len

        return input_ids, token_type_ids, attention_mask

    def convert_text_to_student_feature(self, text, max_seq_len, tokenizer):
        text = [i for i in text]
        input_ids = []
        for i,char in enumerate(text):
            if i < max_seq_len:
                input_ids.append(tokenizer['char2id'].get(char, 1))
        if len(input_ids) < max_seq_len:
            input_ids = input_ids + (max_seq_len - len(input_ids)) * [0]

        assert len(input_ids) == max_seq_len

        return input_ids


PROCESSOR = {
    'KDProcessor': KDProcessor,
}
