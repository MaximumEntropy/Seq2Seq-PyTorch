#!/u/subramas/miniconda2/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys
sys.path.append('/u/subramas/Research/nmt-pytorch/')
from data_utils import *
from model import Seq2Seq, Seq2SeqAttention
import math
import numpy as np

print 'Reading data ...'

src, trg = read_nmt_data(
    '/Tmp/subramas/nmt-pytorch/en-zh/train',
    'cmu-mthomework.train.unk.zh',
    'cmu-mthomework.train.unk.en'
)
'''
src, trg = read_nmt_data(
    '/data/lisatmp4/subramas/datasets/wmt15/deen/train',
    'all_de-en.de.tok.shuf',
    'all_de-en.en.tok.shuf'
)
'''
batch_size = 80
max_length = 80
src_vocab_size = len(src['word2id'])
trg_vocab_size = len(trg['word2id'])

print 'Found %d words in src ' % (src_vocab_size)
print 'Found %d words in trg ' % (trg_vocab_size)

weight_mask = torch.ones(trg_vocab_size).cuda()
weight_mask[trg['word2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)

'''
model = Seq2Seq(
    src_emb_dim=256,
    trg_emb_dim=256,
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_hidden_dim=512,
    trg_hidden_dim=512,
    batch_size=32,
    bidirectional=False,
    nlayers=1,
    dropout=0.,
    peek_dim=0
).cuda()
'''
model = Seq2SeqAttention(
    src_emb_dim=512,
    trg_emb_dim=512,
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_hidden_dim=1024,
    trg_hidden_dim=1024,
    ctx_hidden_dim=1024,
    batch_size=batch_size,
    bidirectional=True,
    nlayers=2,
    dropout=0.,
    peek_dim=0
).cuda()


def clip_gradient(model, clip):
    """Compute a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))

# import ipdb
# ipdb.set_trace()
optimizer = optim.Adadelta(model.parameters())

for i in xrange(1000):
    losses = []
    for j in xrange(0, len(src['data']), batch_size):

        input_lines_src, _, lens_src, mask_src = get_minibatch(
            src, j, batch_size, max_length, add_start=False, add_end=False
        )
        input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(
            trg, j, batch_size, max_length, add_start=False, add_end=False
        )

        if input_lines_src.size()[0] != batch_size:
            break

        decoder_logit = model(input_lines_src, input_lines_trg)
        optimizer.zero_grad()
        loss = loss_criterion(decoder_logit.view(-1, trg_vocab_size), output_lines_trg.view(-1))
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()

        if j % 1000 == 0:
            print 'Epoch : %d Minibatch : %d Loss : %.5f' % (i, j, np.mean(losses))
            losses = []
        if j % 10000 == 0:
            word_probs = model.decode(decoder_logit).data.cpu().numpy().argmax(axis=-1)
            output_lines_trg = output_lines_trg.data.cpu().numpy()
            for sentence_pred, sentence_real in zip(word_probs[:5], output_lines_trg[:5]):
                sentence_pred = [trg['id2word'][x] for x in sentence_pred]
                sentence_real = [trg['id2word'][x] for x in sentence_real]

                index = sentence_real.index('</s>')
                sentence_real = sentence_real[:index]
                sentence_pred = sentence_pred[:index]

                print '---------------------------------------------------'
                print ' '.join(sentence_pred)
                print ' '.join(sentence_real)
                print '---------------------------------------------------'
