''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from Modules import BottleLinear as Linear
from Layers import EncoderLayer, DecoderLayer
import sys
import torch.nn.functional as F

Constants_PAD=0
batch_size=32

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants_PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=3, n_head=3, d_k=64, d_v=64,
            d_word_vec=128, d_model=128, d_inner_hid=128, dropout=0.8):

        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants_PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants_PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.output=nn.Linear(800*128,19)


    def forward(self,src_seq,src_pos,return_attns=False):

        # print(input.size())
        # sys.exit(0)
        # src_seq=input[:800].view(1,-1)
        # src_pos=input[800:].view(1,-1)
        # Word embedding look up
        enc_input = self.src_word_emb(src_seq)

        # Position Encoding addition
        enc_input += self.position_enc(src_pos)
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]
        # print(enc_output.size())
        enc_output=self.output(enc_output.view(-1,800*128))
        enc_output=F.log_softmax(enc_output, dim=1)

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output
