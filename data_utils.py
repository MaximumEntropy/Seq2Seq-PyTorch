import os
import torch
import torchtext
from torch.autograd import Variable
import operator


def construct_vocab(lines, vocab_size):
    """Construct a vocabulary from tokenized lines."""
    vocab = {}
    for line in lines:
        for word in line:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    word2id = {
        '<s>': 0,
        '<pad>': 1,
        '</s>': 2,
        '<unk>': 3,
    }

    id2word = {
        0: '<s>',
        1: '<pad>',
        2: '</s>',
        3: '<unk>',
    }

    sorted_word2id = sorted(
        vocab.items(),
        key=operator.itemgetter(1),
        reverse=True
    )

    sorted_words = [x[0] for x in sorted_word2id[:vocab_size]]

    for ind, word in enumerate(sorted_words):
        word2id[word] = ind + 4

    for ind, word in enumerate(sorted_words):
        id2word[ind + 4] = word

    return word2id, id2word


def read_nmt_data(base_dir, src, trg):
    """Read data from files."""
    src_lines = [line.strip().split() for line in open(os.path.join(base_dir, src))]
    trg_lines = [line.strip().split() for line in open(os.path.join(base_dir, trg))]

    print 'Constructing vocabulary ...'
    src_word2id, src_id2word = construct_vocab(src_lines, 30000)
    trg_word2id, trg_id2word = construct_vocab(trg_lines, 30000)

    src = {'data': src_lines, 'word2id': src_word2id, 'id2word': src_id2word}
    trg = {'data': trg_lines, 'word2id': trg_word2id, 'id2word': trg_id2word}

    return src, trg


def get_minibatch(data_dict, index, batch_size, max_len, add_start=True, add_end=True):
    """Prepare minibatch."""
    lines = data_dict['data']
    word2ind = data_dict['word2id']

    if add_start and add_end:
        lines = [
            ['<s>'] + line + ['</s>']
            for line in lines[index:index + batch_size]
        ]
    elif add_start and not add_end:
        lines = [
            ['<s>'] + line
            for line in lines[index:index + batch_size]
        ]
    elif not add_start and add_end:
        lines = [
            line + ['</s>']
            for line in lines[index:index + batch_size]
        ]
    elif not add_start and not add_end:
        lines = [
            line
            for line in lines[index:index + batch_size]
        ]
    lines = [line[:max_len] for line in lines]

    lens = [len(line) for line in lines]
    max_len = max(lens)

    input_lines = [
        [word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]] +
        [word2ind['<pad>']] * (max_len - len(line))
        for line in lines
    ]

    output_lines = [
        [word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[1:]] +
        [word2ind['<pad>']] * (max_len - len(line))
        for line in lines
    ]

    mask = [
        ([1] * (l - 1)) + ([0] * (max_len - l))
        for l in lens
    ]

    input_lines = Variable(torch.LongTensor(input_lines)).cuda()
    output_lines = Variable(torch.LongTensor(output_lines)).cuda()
    mask = Variable(torch.LongTensor(mask)).cuda()

    return input_lines, output_lines, lens, mask


def get_data():

    src = torchtext.data.Field()
    trg = torchtext.data.Field()

    mt_train = torchtext.datasets.TranslationDataset(
        path='/Tmp/subramas/nmt-pytorch/en-zh/train/',
        exts=('cmu-mthomework.train.unk.en', 'cmu-mthomework.train.unk.zh'),
        fields=(src, trg)
    )
    mt_dev = torchtext.datasets.TranslationDataset(
        path='/Tmp/subramas/nmt-pytorch/en-zh/dev/',
        exts=('cmu-mthomework.dev.unk.en', 'cmu-mthomework.dev.unk.zh'),
        fields=(src, trg)
    )

    src.build_vocab(mt_train, max_size=30000)
    trg.build_vocab(mt_train, max_size=30000)

    train_iter = torchtext.data.BucketIterator(
        mt_train,
        batch_size=32,
        sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg)),
        repeat=False,
    )

    return train_iter, mt_train, mt_dev, src, trg
