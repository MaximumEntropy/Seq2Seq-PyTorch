"""Sequence to Sequence models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from layers import StackedAttentionLSTM


class Seq2Seq(nn.Module):
    """A Vanilla Sequence to Sequence (Seq2Seq) model with LSTMs.

    Ref: Sequence to Sequence Learning with Neural Nets
    https://arxiv.org/abs/1409.3215
    """

    def __init__(
        self, src_emb_dim, trg_emb_dim, src_vocab_size,
        trg_vocab_size, src_hidden_dim, trg_hidden_dim,
        batch_size, pad_token_src, pad_token_trg, bidirectional=True,
        nlayers_src=2, nlayers_trg=1, dropout=0.,
    ):
        """Initialize Seq2Seq Model."""
        super(Seq2Seq, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers_src = nlayers_src
        self.nlayers_trg = nlayers_trg
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg
        self.src_hidden_dim = src_hidden_dim // 2 \
            if self.bidirectional else src_hidden_dim

        self.src_embedding = nn.Embedding(
            num_embedding=src_vocab_size,
            embedding_dim=src_emb_dim,
            padding_idx=self.pad_token_src,
            sparse=True
        )

        self.trg_embedding = nn.Embedding(
            num_embedding=trg_vocab_size,
            embedding_dim=trg_emb_dim,
            padding_idx=self.pad_token_trg,
            sparse=True
        )

        self.encoder = nn.LSTM(
            src_emb_dim,
            self.src_hidden_dim,
            self.nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = nn.LSTM(
            trg_emb_dim,
            trg_hidden_dim,
            nlayers_trg,
            dropout=self.dropout,
            batch_first=True
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )

        self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)
        '''
        nn.init.xavier_normal(
            self.encoder2decoder.weight,
            gain=nn.init.calculate_gain('tanh')
        )
        nn.init.xavier_normal(
            self.decoder2vocab.weight,
            gain=nn.init.calculate_gain('sigmoid')
        )
        '''

    def forward(self, input_src, input_trg, src_lengths):
        r"""Propogate input through the network.

        inputs: input_src, input_trg
        input_src     - batch size x source sequence length x \
            embedding dimension
        input_trg     - batch size x target sequence length x \
            embedding dimension
        src_lengths   - batch size (list)

        returns: decoder_logit (pre-softmax distribution over words)
        decoder_logit - batch size x target sequence length x target vocab size
        """
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        src_emb = pack_padded_sequence(src_emb, src_lengths, batch_first=True)

        _, (src_h_t, src_c_t) = self.encoder(src_emb)

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_init_state = F.tanh(self.encoder2decoder(h_t))

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (
                decoder_init_state.unsqueeze(0).expand(
                    self.decoder.num_layers,
                    decoder_init_state.size(0),
                    decoder_init_state.size(1)
                ),
                c_t.unsqueeze(0).expand(
                    self.decoder.num_layers,
                    c_t.size(0),
                    c_t.size(1)
                )
            )
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1), trg_h.size(2)
        )

        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size(0), trg_h.size(1), decoder_logit.size(1)
        )

        return decoder_logit

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, logits.size(2))
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size(0), logits.size(1), logits.size(2)
        )
        return word_probs


class Seq2SeqAttention(nn.Module):
    """A Sequence to Sequence (Seq2Seq) model with LSTMs and attention.

    Ref: Neural Machine Translation by Jointly Learning to Align and Translate
    https://arxiv.org/abs/1409.0473
    """

    def __init__(
        self, src_emb_dim, trg_emb_dim, src_vocab_size, trg_vocab_size,
        src_hidden_dim, trg_hidden_dim, batch_size, pad_token_src,
        pad_token_trg, bidirectional=True, nlayers_src=2, nlayers_trg=2,
        dropout=0.
    ):
        """Initialize model."""
        super(Seq2SeqAttention, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers_src = nlayers_src
        self.nlayers_trg = nlayers_trg
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            src_emb_dim,
            self.pad_token_src,
            sparse=True
        )
        self.trg_embedding = nn.Embedding(
            trg_vocab_size,
            trg_emb_dim,
            self.pad_token_trg,
            sparse=True
        )

        self.src_hidden_dim = src_hidden_dim // 2 \
            if self.bidirectional else src_hidden_dim

        self.encoder = nn.LSTM(
            src_emb_dim,
            self.src_hidden_dim,
            self.nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = StackedAttentionLSTM(
            trg_emb_dim,
            trg_hidden_dim,
            self.nlayers_trg,
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)
        '''
        nn.init.xavier_normal(
            self.encoder2decoder.weight,
            gain=nn.init.calculate_gain('tanh')
        )
        nn.init.xavier_normal(
            self.decoder2vocab.weight,
            gain=nn.init.calculate_gain('sigmoid')
        )
        '''

    def forward(self, input_src, input_trg, src_lengths):
        r"""Propogate input through the layer.

        inputs: input_src, input_trg
        input_src     - batch size x source sequence length x \
            embedding dimension
        input_trg     - batch size x target sequence length x \
            embedding dimension
        src_lengths   - batch size (list)

        returns: decoder_logit (pre-softmax distribution over words)
        decoder_logit - batch size x target sequence length x target vocab size
        """
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        src_emb = pack_padded_sequence(src_emb, src_lengths, batch_first=True)

        packed_src_h, (src_h_t, src_c_t) = self.encoder(src_emb)
        src_h, _ = pad_packed_sequence(packed_src_h, batch_first=True)

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_init_state = F.tanh(self.encoder2decoder(h_t))

        ctx = src_h.transpose(0, 1)

        trg_h, (_, _), att = self.decoder(
            trg_emb, (decoder_init_state, c_t), ctx
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1), trg_h.size(2)[2]
        )

        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size(0), trg_h.size(1), decoder_logit.size(1)
        )

        return decoder_logit

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, logits.size(2))
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size(0), logits.size(1), logits.size(2)
        )
        return word_probs


class Seq2SeqFastAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
        self, src_emb_dim, trg_emb_dim, src_vocab_size,
        trg_vocab_size, src_hidden_dim, trg_hidden_dim,
        batch_size, pad_token_src, pad_token_trg,
        bidirectional=True, nlayers=2, nlayers_trg=2, dropout=0.
    ):
        """Initialize model."""
        super(Seq2SeqFastAttention, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg

        assert trg_hidden_dim == src_hidden_dim

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            src_emb_dim,
            self.pad_token_src,
            sparse=True
        )
        self.trg_embedding = nn.Embedding(
            trg_vocab_size,
            trg_emb_dim,
            self.pad_token_trg,
            sparse=True
        )

        self.src_hidden_dim = src_hidden_dim // 2 \
            if self.bidirectional else src_hidden_dim

        self.encoder = nn.LSTM(
            src_emb_dim,
            self.src_hidden_dim,
            nlayers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = nn.LSTM(
            trg_emb_dim,
            trg_hidden_dim,
            nlayers_trg,
            batch_first=True,
            dropout=self.dropout
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )

        self.decoder2vocab = nn.Linear(2 * trg_hidden_dim, trg_vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)
        '''
        nn.init.xavier_normal(
            self.encoder2decoder.weight,
            gain=nn.init.calculate_gain('tanh')
        )
        nn.init.xavier_normal(
            self.decoder2vocab.weight,
            gain=nn.init.calculate_gain('sigmoid')
        )
        '''

    def forward(self, input_src, input_trg, src_lengths):
        r"""Propogate input through the network.

        inputs: input_src, input_trg
        input_src     - batch size x source sequence length x \
            embedding dimension
        input_trg     - batch size x target sequence length x \
            embedding dimension
        src_lengths   - batch size (list)

        returns: decoder_logit (pre-softmax distribution over words)
        decoder_logit - batch size x target sequence length x target vocab size
        """
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        src_emb = pack_padded_sequence(src_emb, src_lengths, batch_first=True)
        src_h, (src_h_t, src_c_t) = self.encoder(src_emb)

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_init_state = F.Tanh(self.encoder2decoder(h_t))

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (
                decoder_init_state.unsqueeze(0).expand(
                    self.decoder.num_layers,
                    decoder_init_state.size(0),
                    decoder_init_state.size(1)
                ),
                c_t.unsqueeze(0).expand(
                    self.decoder.num_layers,
                    c_t.size(0),
                    c_t.size(1)
                )
            )
        )

        # Fast Attention

        alpha = torch.bmm(src_h, trg_h.transpose(1, 2))
        alpha = torch.bmm(alpha.transpose(1, 2), src_h)
        trg_h_reshape = torch.cat((trg_h, alpha), 2)

        trg_h_reshape = trg_h_reshape.view(
            trg_h_reshape.size(0) * trg_h_reshape.size(1),
            trg_h_reshape.size(2)
        )

        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.trg_vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs
