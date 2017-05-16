"""Data utilities."""
import torch
from torch.autograd import Variable
import operator
import json
import numpy as np


class Minibatch(object):
    """Parallel data minibatch."""

    def __init__(
        self, src_input, trg_input, trg_output, src_lens
    ):
        """Initialize parameters."""
        self.src_input = src_input
        self.trg_input = trg_input
        self.trg_output = trg_output
        self.src_lens = src_lens


def hyperparam_string(config):
    """Hyerparam string."""
    exp_name = ''
    exp_name += 'model_%s__' % (config['data']['task'])
    exp_name += 'src_%s__' % (config['model']['src_lang'])
    exp_name += 'trg_%s__' % (config['model']['trg_lang'])
    exp_name += 'attention_%s__' % (config['model']['seq2seq'])
    exp_name += 'dim_%s__' % (config['model']['dim'])
    exp_name += 'emb_dim_%s__' % (config['model']['dim_word_src'])
    exp_name += 'optimizer_%s__' % (config['training']['optimizer'])
    exp_name += 'n_layers_src_%d__' % (config['model']['n_layers_src'])
    exp_name += 'n_layers_trg_%d__' % (1)
    exp_name += 'bidir_%s' % (config['model']['bidirectional'])

    return exp_name


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def construct_vocab(lines, vocab_size):
    """Construct a vocabulary from tokenized lines."""
    vocab = {}
    for line in lines:
        for word in line:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    # Discard start, end, pad and unk tokens if already present
    if '<s>' in vocab:
        del vocab['<s>']
    if '<pad>' in vocab:
        del vocab['<pad>']
    if '</s>' in vocab:
        del vocab['</s>']
    if '<unk>' in vocab:
        del vocab['<unk>']

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


def read_dialog_summarization_data(src, config, trg):
    """Read data from files."""
    print('Reading source data ...')
    src_lines = []
    with open(src, 'r') as f:
        for ind, line in enumerate(f):
            src_lines.append(line.strip().split())

    print('Reading target data ...')
    trg_lines = []
    with open(trg, 'r') as f:
        for line in f:
            trg_lines.append(line.strip().split())

    print('Constructing common vocabulary ...')
    word2id, id2word = construct_vocab(
        src_lines + trg_lines, config['data']['n_words_src']
    )

    src = {'data': src_lines, 'word2id': word2id, 'id2word': id2word}
    trg = {'data': trg_lines, 'word2id': word2id, 'id2word': id2word}

    return src, trg


def read_parallel_data(src, trg, config):
    """Read data from files."""
    print('Reading source data ...')
    src_lines = []
    with open(src, 'r') as f:
        for ind, line in enumerate(f):
            src_lines.append(line.strip().split())

    print('Constructing source vocabulary ...')
    src_word2id, src_id2word = construct_vocab(
        src_lines, config['data']['n_words_src']
    )

    src = {'data': src_lines, 'word2id': src_word2id, 'id2word': src_id2word}
    del src_lines

    print('Reading target data ...')
    trg_lines = []
    with open(trg, 'r') as f:
        for line in f:
            trg_lines.append(line.strip().split())

    print('Constructing target vocabulary ...')
    trg_word2id, trg_id2word = construct_vocab(
        trg_lines, config['data']['n_words_trg']
    )

    trg = {'data': trg_lines, 'word2id': trg_word2id, 'id2word': trg_id2word}

    return src, trg


def read_summarization_data(src, trg):
    """Read data from files."""
    src_lines = [line.strip().split() for line in open(src, 'r')]
    trg_lines = [line.strip().split() for line in open(trg, 'r')]
    word2id, id2word = construct_vocab(src_lines + trg_lines, 30000)
    src = {'data': src_lines, 'word2id': word2id, 'id2word': id2word}
    trg = {'data': trg_lines, 'word2id': word2id, 'id2word': id2word}

    return src, trg


def get_parallel_minibatch(
    src_lines, trg_lines, src_word2id, trg_word2id,
    index, batch_size, max_len_src, max_len_trg
):
    """Prepare minibatch."""
    src_lines = [line[:max_len_src] for line in src_lines]
    trg_lines = [line[:max_len_trg] for line in trg_lines]

    src_lens = [len(line) for line in src_lines]
    sorted_indices = np.argsort(src_lens)[::-1]

    sorted_src_lines = [src_lines[idx] for idx in sorted_indices]
    sorted_trg_lines = [trg_lines[idx] for idx in sorted_indices]
    sorted_src_lens = [len(line) for line in sorted_src_lines]
    sorted_trg_lens = [len(line) for line in sorted_trg_lines]

    max_src_len = max(sorted_src_lens)
    max_trg_len = max(sorted_trg_lens)

    input_lines_src = [
        [src_word2id[w] if w in src_word2id else src_word2id['<unk>'] for w in line] +
        [src_word2id['<pad>']] * (max_src_len - len(line))
        for line in sorted_src_lens
    ]

    input_lines_trg = [
        [trg_word2id[w] if w in trg_word2id else trg_word2id['<unk>'] for w in line[:-1]] +
        [trg_word2id['<pad>']] * (max_trg_len - len(line))
        for line in sorted_trg_lines
    ]

    output_lines_trg = [
        [trg_word2id[w] if w in trg_word2id else trg_word2id['<unk>'] for w in line[1:]] +
        [trg_word2id['<pad>']] * (max_trg_len - len(line))
        for line in sorted_trg_lines
    ]

    input_lines_src = Variable(torch.LongTensor(input_lines_src)).cuda()
    input_lines_trg = Variable(torch.LongTensor(input_lines_trg)).cuda()
    output_lines_trg = Variable(torch.LongTensor(output_lines_trg)).cuda()

    return Minibatch(
        input_lines_src, input_lines_trg,
        output_lines_trg, sorted_src_lens
    )


def get_minibatch(
    lines, word2ind, index, batch_size,
    max_len, add_start=True, add_end=True
):
    """Prepare minibatch."""
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
    mask = Variable(torch.FloatTensor(mask)).cuda()

    return input_lines, output_lines, lens, mask


def get_autoencode_minibatch(
    lines, word2ind, index, batch_size,
    max_len, add_start=True, add_end=True
):
    """Prepare minibatch."""
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
        ([1] * (l)) + ([0] * (max_len - l))
        for l in lens
    ]

    input_lines = Variable(torch.LongTensor(input_lines)).cuda()
    output_lines = Variable(torch.LongTensor(output_lines)).cuda()
    mask = Variable(torch.FloatTensor(mask)).cuda()

    return input_lines, output_lines, lens, mask
