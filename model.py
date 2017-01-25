import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F


class RNNCellBase(nn.Module):
    """RNN Cell Base Class."""

    def __repr__(self):
        """Way to display cell."""
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias != True:
            s += ', bias={bias}}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LSTMAttention(RNNCellBase):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, context_size):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = 1

        self.input_weights_1 = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.hidden_weights_1 = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.input_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hidden_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.input_weights_2 = nn.Parameter(torch.Tensor(4 * hidden_size, context_size))
        self.hidden_weights_2 = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.input_bias_2 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hidden_bias_2 = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.context2attention = nn.Parameter(torch.Tensor(context_size, context_size))
        self.bias_context2attention = nn.Parameter(torch.Tensor(context_size))

        self.hidden2attention = nn.Parameter(torch.Tensor(context_size, hidden_size))

        self.input2attention = nn.Parameter(torch.Tensor(input_size, context_size))

        self.recurrent2attention = nn.Parameter(torch.Tensor(context_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        stdv_ctx = 1.0 / math.sqrt(self.context_size)

        self.input_weights_1.data.uniform_(-stdv, stdv)
        self.hidden_weights_1.data.uniform_(-stdv, stdv)
        self.input_bias_1.data.fill_(0)
        self.hidden_bias_1.data.fill_(0)

        self.input_weights_2.data.uniform_(-stdv_ctx, stdv_ctx)
        self.hidden_weights_2.data.uniform_(-stdv, stdv)
        self.input_bias_2.data.fill_(0)
        self.hidden_bias_2.data.fill_(0)

        self.context2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.bias_context2attention.data.fill_(0)

        self.hidden2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.input2attention.data.uniform_(-stdv_ctx, stdv_ctx)

        self.recurrent2attention.data.uniform_(-stdv_ctx, stdv_ctx)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden, projected_input, projected_ctx):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim

            gates = F.linear(
                input, self.input_weights_1, self.input_bias_1
            ) + F.linear(hx, self.hidden_weights_1, self.hidden_bias_1)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            # Attention mechanism

            # Project current hidden state to context size
            hidden_ctx = F.linear(hy, self.hidden2attention)
            # print 'Hidden context', hidden_ctx.size()

            # Added projected hidden state to each projected context
            hidden_ctx_sum = projected_ctx + hidden_ctx.unsqueeze(0).expand(
                projected_ctx.size()
            )
            # print 'Summed context with hidden state ', hidden_ctx_sum.size()

            # Add this to projected input at this time step
            hidden_ctx_sum = hidden_ctx_sum + \
                projected_input.unsqueeze(0).expand(hidden_ctx_sum.size())

            # Non-linearity
            hidden_ctx_sum = F.tanh(hidden_ctx_sum)

            # Compute alignments
            alpha = torch.bmm(
                hidden_ctx_sum,
                self.recurrent2attention.unsqueeze(0).expand(
                    hidden_ctx_sum.size(0),
                    self.recurrent2attention.size(0),
                    self.recurrent2attention.size(1)
                )
            ).squeeze().t()
            alpha = F.softmax(alpha).t()
            weighted_context = torch.mul(
                projected_ctx, alpha.unsqueeze(2).expand(projected_ctx.size())
            ).sum(0).squeeze()

            gates = F.linear(
                weighted_context, self.input_weights_2, self.input_bias_2
            ) + F.linear(hy, self.hidden_weights_2, self.hidden_bias_2)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cy) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            return hy, cy

        projected_ctx = torch.bmm(
            ctx,
            self.context2attention.unsqueeze(0).expand(
                ctx.size(0),
                self.context2attention.size(0),
                self.context2attention.size(1)
            ),
        )
        projected_ctx += self.bias_context2attention.unsqueeze(0).unsqueeze(0).expand(
            projected_ctx.size()
        )

        projected_input = torch.bmm(
            input,
            self.input2attention.unsqueeze(0).expand(
                input.size(0),
                self.input2attention.size(0),
                self.input2attention.size(1)
            ),
        )
        # print 'Projected input ', projected_input.size()
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(
                input[i], hidden, projected_input[i], projected_ctx
            )
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return output, hidden


class LSTMAttentionDot(RNNCellBase):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size):
        """Initialize params."""
        super(LSTMAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.input_weights_1 = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.hidden_weights_1 = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.input_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hidden_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.Wc = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 2))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        stdv = 1.0 / math.sqrt(self.hidden_size)

        self.input_weights_1.data.uniform_(-stdv, stdv)
        self.hidden_weights_1.data.uniform_(-stdv, stdv)
        self.input_bias_1.data.fill_(0)
        self.hidden_bias_1.data.fill_(0)

        self.Wc.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim

            gates = F.linear(input, self.input_weights_1, self.input_bias_1) + \
                F.linear(hx, self.hidden_weights_1, self.hidden_bias_1)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            # Attention mechanism
            alpha = torch.bmm(
                hy.unsqueeze(1),
                ctx.view(ctx.size(1), ctx.size(2), ctx.size(0))
            )
            alpha = F.softmax(alpha.squeeze()).t()
            weighted_context = torch.mul(ctx, alpha.unsqueeze(2).expand(
                ctx.size()
            )).sum(0).squeeze()
            h_tilde = torch.cat((weighted_context, hy), 1)
            h_tilde = F.tanh(F.linear(h_tilde, self.Wc))

            return h_tilde, cy

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return output, hidden


class Seq2Seq(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
        self,
        src_emb_dim,
        trg_emb_dim,
        src_vocab_size,
        trg_vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        batch_size,
        bidirectional=True,
        nlayers=2,
        dropout=0.,
        peek_dim=0
    ):
        """Initialize model."""
        super(Seq2Seq, self).__init__()
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
        self.peek_dim = peek_dim
        self.src_embedding = nn.Embedding(src_vocab_size, src_emb_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, trg_emb_dim)

        self.encoder = nn.LSTM(
            src_emb_dim,
            src_hidden_dim,
            nlayers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            trg_emb_dim + peek_dim,
            trg_hidden_dim,
            1,
            batch_first=True
        )

        self.encoder2decoder = nn.Linear(
            src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size).cuda()
        if self.peek_dim > 0:
            self.encoder2peek = nn.Linear(
                src_hidden_dim * self.num_directions,
                peek_dim
            )
        self.h0_encoder = nn.Parameter(torch.randn(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            src_hidden_dim
        ))
        self.c0_encoder = nn.Parameter(torch.randn(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            src_hidden_dim
        ))
        self.register_parameter('h0_encoder', self.h0_encoder)
        self.register_parameter('c0_encoder', self.c0_encoder)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def forward(self, input_src, input_trg):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]
        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        if self.peek_dim > 0:
            projected_src = nn.Tanh()(self.encoder2peek(h_t))
            projected_src = Variable(projected_src.data.repeat(1, trg_emb.size()[1]).view(
                projected_src.size()[0], trg_emb.size()[1], projected_src.size()[1]
            ))
            trg_emb = torch.cat((trg_emb, projected_src), 2)

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (
                decoder_init_state.view(
                    self.decoder.num_layers,
                    decoder_init_state.size()[0],
                    decoder_init_state.size()[1]
                ),
                c_t.view(
                    self.decoder.num_layers,
                    c_t.size()[0],
                    c_t.size()[1]
                )
            )
        )
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
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


class Seq2SeqAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
        self,
        src_emb_dim,
        trg_emb_dim,
        src_vocab_size,
        trg_vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        ctx_hidden_dim,
        batch_size,
        bidirectional=True,
        nlayers=2,
        dropout=0.,
        peek_dim=0
    ):
        """Initialize model."""
        super(Seq2SeqAttention, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.ctx_hidden_dim = ctx_hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.peek_dim = peek_dim
        self.src_embedding = nn.Embedding(src_vocab_size, src_emb_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, trg_emb_dim)
        src_hidden_dim = src_hidden_dim // 2 if self.bidirectional else src_hidden_dim
        self.encoder = nn.LSTM(
            src_emb_dim,
            src_hidden_dim,
            nlayers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.decoder = LSTMAttention(
            trg_emb_dim,
            trg_hidden_dim,
            ctx_hidden_dim
        )
        self.encoder2decoder = nn.Linear(
            src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size)
        if self.peek_dim > 0:
            self.encoder2peek = nn.Linear(
                src_hidden_dim * self.num_directions,
                peek_dim
            )
        self.h0_encoder = nn.Parameter(torch.randn(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            src_hidden_dim
        ))
        self.c0_encoder = nn.Parameter(torch.randn(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            src_hidden_dim
        ))
        self.register_parameter('h0_encoder', self.h0_encoder)
        self.register_parameter('c0_encoder', self.c0_encoder)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def forward(self, input_src, input_trg):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]
        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        if self.peek_dim > 0:
            projected_src = nn.Tanh()(self.encoder2peek(h_t))
            projected_src = Variable(projected_src.data.repeat(1, trg_emb.size()[1]).view(
                projected_src.size()[0], trg_emb.size()[1], projected_src.size()[1]
            ))
            trg_emb = torch.cat((trg_emb, projected_src), 2)

        ctx = src_h.contiguous().view(
            src_h.size()[1], src_h.size()[0], src_h.size()[2]
        )
        # print ctx.size(), decoder_init_state.size(), c_t.size()
        trg_h, (_, _) = self.decoder(
            trg_emb.view(
                trg_emb.size()[1], trg_emb.size()[0], trg_emb.size()[2]
            ),
            (decoder_init_state, c_t),
            ctx
        )
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
        )
        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[1],
            trg_h.size()[0],
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
