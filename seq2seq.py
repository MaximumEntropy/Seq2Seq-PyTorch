#!/usr/bin/env/ python

"""Main script to run things"""
from data_utils import read_parallel_data, read_config, \
    hyperparam_string, get_parallel_minibatch
from model import Seq2Seq, Seq2SeqAttention, Seq2SeqFastAttention
from evaluate import evaluate_model
import numpy as np
import logging
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
args = parser.parse_args()
config_file_path = args.config
config = read_config(config_file_path)
experiment_name = hyperparam_string(config)
save_dir = config['data']['save_dir']
load_dir = config['data']['load_dir']
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (experiment_name),
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


logging.info('Reading training data ...')
src, trg = read_parallel_data(
    src=config['data']['src'],
    trg=config['data']['trg'],
    config=config
)

logging.info('Reading validation data ...')
src_valid, trg_valid = read_parallel_data(
    src=config['data']['valid_src'],
    trg=config['data']['valid_trg'],
    config=config
)

logging.info('Reading test data ...')
src_test, trg_test = read_parallel_data(
    src=config['data']['test_src'],
    trg=config['data']['test_trg'],
    config=config
)

batch_size = config['data']['batch_size']
max_length_src = config['data']['max_src_length']
max_length_trg = config['data']['max_trg_length']
src_vocab_size = len(src['word2id'])
trg_vocab_size = len(trg['word2id'])

logging.info('###################################')
logging.info('Model Parameters : ')
logging.info('Task : %s ' % (config['data']['task']))
logging.info('Model : %s ' % (config['model']['seq2seq']))
logging.info('Source Language : %s ' % (config['model']['src_lang']))
logging.info('Target Language : %s ' % (config['model']['trg_lang']))
logging.info('Source Word Embedding Dim  : %s' % (
    config['model']['dim_word_src'])
)
logging.info('Target Word Embedding Dim  : %s' % (
    config['model']['dim_word_trg'])
)
logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Target RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
logging.info('Target RNN Depth : %d ' % (config['model']['n_layers_trg']))
logging.info('Source RNN Bidirectional  : %s' % (
    config['model']['bidirectional'])
)
logging.info('Batch Size : %d ' % (config['model']['n_layers_trg']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lrate']))

logging.info('Found %d words in src ' % (src_vocab_size))
logging.info('Found %d words in trg ' % (trg_vocab_size))
logging.info('###################################')

weight_mask = torch.ones(trg_vocab_size).cuda()
weight_mask[trg['word2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()

if config['model']['seq2seq'] == 'vanilla':

    model = Seq2Seq(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        batch_size=batch_size,
        bidirectional=config['model']['bidirectional'],
        pad_token_src=src['word2id']['<pad>'],
        pad_token_trg=trg['word2id']['<pad>'],
        nlayers_src=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.,
    ).cuda()

elif config['model']['seq2seq'] == 'attention':

    model = Seq2SeqAttention(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        batch_size=batch_size,
        bidirectional=config['model']['bidirectional'],
        pad_token_src=src['word2id']['<pad>'],
        pad_token_trg=trg['word2id']['<pad>'],
        nlayers_src=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.,
    ).cuda()

elif config['model']['seq2seq'] == 'fastattention':

    model = Seq2SeqFastAttention(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        batch_size=batch_size,
        bidirectional=config['model']['bidirectional'],
        pad_token_src=src['word2id']['<pad>'],
        pad_token_trg=trg['word2id']['<pad>'],
        nlayers_src=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.,
    ).cuda()

if load_dir == 'auto':
    checkpoints = os.listdir(save_dir)
    epoch_max = -1
    minibatch_max = -1
    load_checkpoint = ''
    for checkpoint in checkpoints:
        checkpoint = checkpoint.split('__')
        epoch = int(checkpoint[-2].split('_')[1])
        minibatch = int(checkpoint[-1].split('_')[1].replace('.model', ''))
        if epoch > epoch_max:
            load_checkpoint = '__'.join(checkpoint)
            epoch_max = epoch_max
            minibatch_max = minibatch_max
        elif epoch == epoch_max and minibatch > minibatch_max:
            load_checkpoint = '__'.join(checkpoint)
            minibatch_max = minibatch_max

    if load_checkpoint != '':
        logging.info('Loading last saved model : %s ' % (load_checkpoint))
        model.load_state_dict(torch.load(
            open(os.path.join(save_dir, load_checkpoint))
        ))

elif load_dir and not load_dir == 'auto':
    logging.info('Loading model : %s ' % (load_dir))
    model.load_state_dict(torch.load(
        open(load_dir)
    ))

logging.info('###################################')
logging.info('Model Architecture')
logging.info(model)
logging.info('###################################')

if config['training']['optimizer'] == 'adam':
    lr = config['training']['lrate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'adadelta':
    optimizer = optim.Adadelta(model.parameters())
elif config['training']['optimizer'] == 'sgd':
    lr = config['training']['lrate']
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    raise NotImplementedError("Learning method not recommend for task")

for i in range(1000):
    losses = []
    for j in range(0, len(src['data']), batch_size):

        minibatch = get_parallel_minibatch(
            src['data'], trg['data'], src['word2id'], trg['word2id'],
            j, batch_size, max_length_src, max_length_trg
        )

        decoder_logit = model(
            minibatch['input_src'],
            minibatch['input_trg'],
            minibatch['src_lens']
        )
        optimizer.zero_grad()

        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, trg_vocab_size),
            minibatch['output_trg'].view(-1)
        )
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()

        if j % config['management']['monitor_loss'] == 0:
            logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (
                i, j, np.mean(losses))
            )
            losses = []

        if (
            config['management']['print_samples'] and
            j % config['management']['print_samples'] == 0
        ):
            word_probs = model.decode(
                decoder_logit
            ).data.cpu().numpy().argmax(axis=-1)

            trg_output = minibatch['output_trg'].data.cpu().numpy()

            for sentence_pred, sentence_real in zip(
                word_probs[:5], trg_output[:5]
            ):
                sentence_pred = [trg['id2word'][x] for x in sentence_pred]
                sentence_real = [trg['id2word'][x] for x in sentence_real]

                if '</s>' in sentence_real:
                    index = sentence_real.index('</s>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]

                logging.info('Predicted : %s ' % (' '.join(sentence_pred)))
                logging.info('-----------------------------------------------')
                logging.info('Real : %s ' % (' '.join(sentence_real)))
                logging.info('===============================================')

        if j % config['management']['checkpoint_freq'] == 0:

            logging.info('Evaluating model on the validation set ...')
            bleu = evaluate_model(
                model, src, src_valid, trg,
                trg_valid, config, verbose=False,
                metric='bleu',
            )
            logging.info('Epoch : %d Minibatch : %d : Valid BLEU : %.5f ' % (
                i, j, bleu)
            )

            logging.info('Evaluating model on the test set ...')
            bleu = evaluate_model(
                model, src, src_test, trg,
                trg_test, config, verbose=False,
                metric='bleu',
            )
            logging.info('Epoch : %d Minibatch : %d : Test BLEU : %.5f ' % (
                i, j, bleu)
            )

            logging.info('Saving model ...')

            torch.save(
                model.state_dict(),
                open(os.path.join(
                    save_dir,
                    experiment_name + '__epoch_%d__minibatch_%d.model' % (i, j)
                ), 'wb'
                )
            )

    bleu = evaluate_model(
        model, src, src_test, trg,
        trg_test, config, verbose=False,
        metric='bleu',
    )

    logging.info('Epoch : %d : BLEU : %.5f ' % (i, bleu))

    torch.save(
        model.state_dict(),
        open(os.path.join(
            save_dir,
            experiment_name + '__epoch_%d__minibatch_%d' % (i, j) + '.model'), 'wb'
        )
    )
