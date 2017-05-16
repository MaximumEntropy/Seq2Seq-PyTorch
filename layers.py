"""Recurrent layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedAttentionLSTM(nn.Module):
    """Stacked LSTMs with Attention."""

    def __init__(self, input_dim, hidden_dim, num_layers):
        """Initialize params."""
        super(StackedAttentionLSTM, self).__init__()
        self.input_size = input_dim
        self.rnn_size = hidden_dim
        self.num_layers = num_layers

        # Stack RNNs on top of each other.
        self.layers = []
        for i in range(num_layers):
            layer = LSTMAttention(input_dim, hidden_dim)
            self.layers.append(layer)
            input_dim = hidden_dim

        self.rnns = nn.ModuleList(self.layers)

    def forward(self, input, hidden, ctx):
        """Propogate input through the layer.

        Run a stack of LSTMs with attention through a sequence of
        embeddings conditioned on another sequence of embeddings.

        inputs:
        input  - batch size x target sequence length  x embedding dimension
        hidden - batch size x hidden dimension
        ctx    - batch size x context sequence length x hidden dimension

        returns: h, (h_t, c_t)
        h      - batch size x target sequence length  x hidden dimension
        h_t    - depth      x batch size              x hidden dimension
        """
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for idx, rnn in enumerate(self.rnns):
            output, (h_1_i, c_1_i), att = rnn(input, (h_0, c_0), ctx)
            input = output

            if idx != len(self.layers):
                input = self.dropout(input)

            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1), att


class DeepBidirectionalLSTM(nn.Module):
    """A Deep LSTM with the first layer being bidirectional."""

    def __init__(
        self, input_dim, hidden_dim,
        num_layers, dropout
    ):
        """Initialize params."""
        super(DeepBidirectionalLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        self.bilstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim // 2,
            1,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout
        )

        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            self.num_layers - 1,
            bidirectional=False,
            batch_first=True,
            dropout=self.dropout
        )

    def forward(self, input):
        """Propogate input forward through the layer.

        inputs:
        input - batch size x sequence length x input dimension

        returns: h, (h_t, c_t)
        h     - batch size x sequence length x hidden dimension
        h_t   - batch size x hidden dimension
        c_t   - batch size x hidden dimension
        """
        h, (_, _) = self.bilstm(input)
        return self.lstm(h)


class SoftAttention(nn.Module):
    """Soft Attention Layer.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT - https://github.com/OpenNMT/OpenNMT-py.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, input, ctx):
        """Propogate input through the layer.

        inputs:
        input   - batch size x dim
        ctx     - batch size x context sequence length x dim

        returns: h_tilde, attn
        h_tilde - batch size x dim
        attn    - batch size x context sequence length
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(ctx, target).squeeze(2)  # batch x sourceL
        attn = F.softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, ctx).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = F.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LSTMAttention(nn.Module):
    """A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_dim, hidden_dim):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 1

        self.input_weights = nn.Linear(self.input_dim, 4 * self.hidden_dim)
        self.hidden_weights = nn.Linear(self.hidden_dim, 4 * self.hidden_dim)

        self.attention_layer = SoftAttention(self.hidden_dim)

    def forward(self, input, hidden, ctx):
        r"""Propogate input through the layer.

        inputs:
        input   - batch size x target sequence length  x embedding dimension
        hidden  - batch size x hidden dimension
        ctx     - batch size x source sequence length  x hidden dimension

        returns: output, hidden
        output  - batch size x target sequence length  x hidden dimension
        hidden  - (batch size x hidden dimension, \
            batch size x hidden dimension)
        """
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return (h_tilde, cy), alpha

        input = input.transpose(0, 1)

        output = []
        alphas = []
        steps = range(input.size(0))
        for i in steps:
            hidden, alpha = recurrence(input[i], hidden)
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)
            alphas.append(alpha)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        alphas = torch.cat(alphas, 0)
        output = output.transpose(0, 1)
        alphas = alphas.transpose(0, 1)
        return output, hidden, alphas
