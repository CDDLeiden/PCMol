import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


## Masking
def padding_mask(seq, pad_idx=0, two_dim=False):
    """
    For masking out the padding part of key sequence.
    """
    ## For 2D sequences (AF embeddings)
    if two_dim:
        return seq.mean(axis=2) == pad_idx
    else:
        return seq == pad_idx


def triangular_mask(seq, diag=1):
    """
    For masking out the subsequent info in the transformer decoder.
    """
    _, len_s = seq.size()
    masks = torch.ones((len_s, len_s)).triu(diagonal=diag)
    masks = masks.bool().to(seq.device)
    return masks


class MeanPooling(nn.Module):
    """
    Pooling layer for the regression heads
    Concatenates the last hidden state of the SMILES and AF2 embeddings
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attn_mask):
        input_mask_expanded = (
            attn_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class PositionwiseFeedForward(nn.Module):
    """
    A two-layer feed-forward-layer module
    Used in the transformer encoder and decoder blocks
    """

    def __init__(self, d_input, d_hidden, regression=False):
        super().__init__()
        outgoing_nodes = 1 if regression else d_input
        self.linear1 = nn.Linear(d_input, d_hidden)
        self.linear2 = nn.Linear(d_hidden, outgoing_nodes)

    def forward(self, x):
        y = self.linear1(x).relu()
        y = self.linear2(y)
        return y


class MLP(nn.Module):
    """
    A multi-layer perceptron module
    Layer sizes are specified as a list of integers
    (including input and output sizes)
    Used in the pchembl auxiliary loss
    """

    def __init__(self, layer_sizes, activation=nn.ReLU(), dropout=0.1):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PreNorm(nn.Module):
    """
    Pre-normalization layer (norm -> dropout -> layer)
    Improvement over the original paper
    """

    def __init__(self, size, dropout=0.1):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer, save=False, **kwargs):
        """
        x: input
        layer: layer to apply
        save: whether to save attention weights
        **kwargs: additional arguments for the layer
        """
        y = self.norm(x)
        y = layer(x, **kwargs)

        ## For saving attention weights
        if save:
            layer.attention_weights = y[1]

        y = self.dropout(y[0])
        return x + y


class CustomDecoderBlock(nn.Module):
    """
    A decoder transformer block module with attention for both the SMILES and AF2 embeddings
    Uses pre-norm structure (norm -> dropout -> layer)
    """

    def __init__(self, d_model, d_emb, n_heads, d_feedforward, dropout=0.1, **kwargs):
        super(CustomDecoderBlock, self).__init__()
        self.smiles_attention = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.alphafold_attention = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_feedforward)
        self.norm = nn.ModuleList(
            [
                PreNorm(d_model, dropout),
                PreNorm(d_model, dropout),
                PreNorm(d_model, dropout),
            ]
        )
        self.attention_weights = []

    def forward(self, x, mem, key_padding_mask=None, encoder_mask=None, attn_mask=None):
        x = self.norm[0](
            x,
            self.smiles_attention,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.norm[1](
            x,
            self.alphafold_attention,
            True,
            key=mem,
            value=mem,
            key_padding_mask=encoder_mask,
        )
        x = self.norm[2](x, self.pos_ffn)
        return x
