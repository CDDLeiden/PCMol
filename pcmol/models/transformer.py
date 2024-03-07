import torch
import torch.nn as nn
from .components import CustomDecoderBlock, MLP, MeanPooling
from .components import padding_mask, triangular_mask


def count_parameters(model):
    """
    Return the number of trainable parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TransformerDecoder(nn.Module):
    """
    Transformer decoder module

    Args:
        vocab (Vocab): Vocab object for SMILES (see utils/vocab.py)
        d_model (int): Dimensionality of the model
        d_emb (int): Dimensionality of the embeddings
        n_heads (int): Number of attention heads
        d_feedforward (int): Dimensionality of the feedforward layer
        n_layers (int): Number of decoder layers
        **kwargs: Additional arguments for the CustomDecoderBlock
    """

    def __init__(
        self,
        vocab,
        d_model=1024,
        d_emb=384,
        n_heads=16,
        d_feedforward=2048,
        n_layers=12,
        **kwargs
    ):

        super(TransformerDecoder, self).__init__()
        self.n_layers = n_layers
        self.d_emb = d_emb
        self.d_model = d_model
        self.vocab = vocab

        if d_emb != d_model:
            # Project the AF2 embeddings to the same dimension as the SMILES embeddings
            self.af_emb_projection = nn.Linear(d_emb, d_model)

        self.token_embedding = nn.Embedding(vocab.size, d_model)
        self.posit_embedding = nn.Embedding(vocab.max_len, d_model)
        blocks = [
            CustomDecoderBlock(
                d_model, d_emb, n_heads, d_feedforward=d_feedforward, **kwargs
            )
            for _ in range(n_layers)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.token_projection = nn.Linear(d_model, self.vocab.size)

    def forward(
        self, x, af_emb, key_padding_mask=None, attn_mask=None, encoder_mask=None
    ):
        """
        Args:
            x: Input sequence tensor of shape (batch_size, seq_len)
            af_emb: Memory tensor of shape (batch_size, seq_len, d_model)
            key_padding_mas): Mask for padding tokens of shape (batch_size, seq_len)
            attn_mask: Attention mask of shape (batch_size, seq_len, seq_len)
            encoder_mask: Encoder mask of shape (batch_size, seq_len, seq_len)
        """
        x = self.posit_embedding(x) + self.token_embedding(x)

        # Project the AF2 embeddings to the same dimension as the SMILES embeddings
        if self.d_emb != self.d_model:
            af_emb = self.af_emb_projection(af_emb)

        for block in self.blocks:
            x = block(
                x,
                mem=af_emb,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                encoder_mask=encoder_mask,
            )
        projection = self.token_projection(x)
        return projection, x


class AF2SmilesTransformer(nn.Module):
    """
    Generative transformer af_emb_projection-decoder model that is conditioned for a specific target using
    AlphaFold2 embeddings
    Extra head for pChEMBL value prediction
    """

    def __init__(
        self,
        vocab,
        d_emb=384,
        d_model=1024,
        n_heads=16,
        d_feedforward=2048,
        n_layers=16,
        dropout=0.1,
        padding_idx=0,
        dev="cuda",
        loss_coefficients={"generative": 1.0, "pchembl": 0.5},
        use_pooling=False,
        pcm=False,
        pchembl_arch=[],
        **kwargs
    ):

        super(AF2SmilesTransformer, self).__init__()
        self.dev = dev
        self.vocab = vocab
        self.padding_idx = padding_idx
        self.decoder = TransformerDecoder(
            self.vocab,
            d_emb=d_emb,
            d_model=d_model,
            dropout=dropout,
            n_heads=n_heads,
            d_feedforward=d_feedforward,
            n_layers=n_layers,
            **kwargs
        )

        # Extra head for Proteochemometric modelling
        self.pooler = None
        if pcm:
            if use_pooling:
                self.pooler = MeanPooling()
                n_input = d_model
            else:
                n_input = d_model * self.vocab.max_len
            layer_sizes = [n_input] + pchembl_arch + [1]
            self.pchembl_projection = MLP(layer_sizes=layer_sizes, dropout=dropout)
            self.pchembl_proj_loss = torch.nn.MSELoss()

        self.loss_coefficients = loss_coefficients
        self.init_states()

    def init_states(self):
        """
        Same initialization as in PyTorch implementation of the Transformer model
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)
            self.to(self.dev)

    def forward(self, x, af_emb, train=False, pchembl_targets=None, calc_pchembl=False):
        """
        Args:
            x: decoder input (SMILES tokens)
            af_emb: AF embeddings
            train: bool flag that indicates whether the model is in training mode
            pchembl_targets: pchembl values for each molecule
            calc_pchembl: whether to calculate pchembl loss
        """
        outputs = {}
        loss = torch.tensor(0.0).to(self.dev)
        losses = dict(generative=0, pchembl=0, total=0)

        batch_size = len(x)
        z = torch.zeros(batch_size, 1).long().to(self.dev)
        key_padding_mask = padding_mask(z)  # Placeholder mask
        # Generation
        if not train:
            # Create empty decoder output tensor
            decoder_output = torch.zeros(len(x), self.vocab.max_len).long().to(self.dev)
            decoder_output[:, : x.size(1)] = x
            is_end = torch.zeros(len(x)).bool().to(self.dev)

            for step in range(self.vocab.max_len - 1):
                decoder_input = decoder_output[:, : x.size(1) + step]  # Offset by 1

                ## Padding masks for AF embeddings and SMILES
                key_padding_mask = padding_mask(decoder_input, self.padding_idx)
                encoder_mask = padding_mask(af_emb, self.padding_idx, two_dim=True)

                # Causal attention mask
                attn_mask = triangular_mask(decoder_input)

                dec, z = self.decoder(
                    decoder_input.transpose(0, 1),
                    af_emb=af_emb.transpose(0, 1),
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    encoder_mask=encoder_mask,
                )

                token = dec.softmax(dim=-1)[-1, :, :].multinomial(1).view(-1)
                token[is_end] = self.vocab.tk2ix["_"]

                # Check if EOS token has been generated
                is_end |= token == self.vocab.tk2ix["EOS"]
                decoder_output[:, x.size(1) + step] = token
                if is_end.all():
                    break
            target = decoder_output

        # Training
        else:
            # Create and prepend 'GO' tokens to decoder input
            bos_token = self.vocab.tk2ix["GO"]
            init_tokens = torch.LongTensor([[bos_token]] * batch_size).to(self.dev)
            decoder_input = torch.cat([init_tokens, x[:, :-1]], dim=1)
            target = x

            key_padding_mask = padding_mask(decoder_input, self.padding_idx)
            encoder_mask = padding_mask(af_emb, self.padding_idx, two_dim=True)
            attn_mask = triangular_mask(decoder_input)

            proj, z = self.decoder(
                decoder_input.transpose(0, 1),
                af_emb=af_emb.transpose(0, 1),
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                encoder_mask=encoder_mask,
            )

            decoder_output = proj.transpose(0, 1).log_softmax(dim=-1)
            loss = decoder_output.gather(2, target.unsqueeze(2)).squeeze(2)
            loss = sum([-l.mean() for l in loss])
            losses["generative"] = loss

        outputs["tokens"] = decoder_output

        ###################
        # Code for property prediction heads + losses
        if calc_pchembl:
            z = z.transpose(0, 1)
            if self.pooler is not None:
                z = self.pooler(z, attn_mask=key_padding_mask)
            else:
                mask = key_padding_mask.unsqueeze(-1).expand(z.size()).float()
                z = z * mask
                z = z.reshape(z.shape[0], -1)

            pchembl_pred = self.pchembl_projection(z).squeeze()
            outputs["pchembl"] = pchembl_pred

            if pchembl_targets is not None:
                p_loss = self.pchembl_proj_loss(pchembl_pred, pchembl_targets)
                p_loss *= self.loss_coefficients["pchembl"]
                loss += p_loss.item()
                losses["pchembl"] = p_loss

        losses["total"] = loss

        return decoder_output, loss
