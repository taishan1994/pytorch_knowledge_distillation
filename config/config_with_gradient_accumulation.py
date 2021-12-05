class BertForClassificationConfig:
    data_dir = './data/THUCNews/'
    model_dir = '../../model_hub/bert-base-chinese/'
    save_dir = './checkpoints/'

    with open('./data/THUCNews/cnews.labels', 'r') as fp:
        labels = fp.read().strip().split('\n')
    label2id = {}
    id2label = {}
    for k, v in enumerate(labels):
        label2id[v] = k
        id2label[k] = v

    do_train = True
    do_eval = True
    do_test = True
    do_predict = True

    use_accumulation_steps = True
    accumulation_steps = 4

    max_seq_len = 256
    epochs = 10
    weight_decay = 0.01
    lr = 8e-5
    adam_epsilon = 1e-8
    warmup_proporation = 0.1
    train_batch_size = 32
    eval_batch_size = 32


class BilstmForClassificationConfig:
    data_dir = './data/THUCNews/'
    save_dir = './checkpoints/'

    with open('./data/THUCNews/cnews.labels', 'r') as fp:
        labels = fp.read().strip().split('\n')
    label2id = {}
    id2label = {}
    for k, v in enumerate(labels):
        label2id[v] = k
        id2label[k] = v

    with open('./data/THUCNews/cnews.vocab.txt', 'r') as fp:
        vocab = fp.read().strip().split('\n')
    vocab_size = len(vocab)

    do_train = True
    do_eval = True
    do_test = True
    do_predict = True

    use_accumulation_steps = True
    accumulation_steps = 4

    max_seq_len = 256
    word_embedding_dimension = 300
    hidden_dim = 384
    epochs = 10
    weight_decay = 0.01
    lr = 8e-5
    adam_epsilon = 1e-8
    warmup_proporation = 0.1
    train_batch_size = 32
    eval_batch_size = 32
