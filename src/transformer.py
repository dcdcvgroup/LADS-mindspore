"""
DETR Transformer class.
from nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import context
from mindspore.common.initializer import initializer, HeNormal, HeUniform, One, Zero, XavierUniform, XavierNormal
from src.init_weights import KaimingUniform, UniformBias

class MultiHeadAttention(nn.Cell):

    def __init__(self, d_model, heads, dropout=0.1, has_mask=True):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.has_mask = has_mask

        self.q_dense = nn.Dense(d_model, d_model, weight_init=initializer('xavier_uniform', [d_model, d_model]))
        self.k_dense = nn.Dense(d_model, d_model, weight_init=initializer('xavier_uniform', [d_model, d_model]))
        self.v_dense = nn.Dense(d_model, d_model, weight_init=initializer('xavier_uniform', [d_model, d_model]))
        # =====================
        # self.in_proj = nn.Dense(3*d_model,d_model,
        #                         weight_init=initializer('xavier_uniform', [d_model,3*d_model],ms.float16))
        # =====================
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.out_proj = nn.Dense(d_model, d_model, weight_init=initializer('xavier_uniform', [d_model, d_model]))

        self.softmax = ops.Softmax(axis=-1)
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.batch_mul = ops.BatchMatMul()
        self.expand_dims = ops.ExpandDims()
        self.sqrt = ops.Sqrt()
        self.ones_like = ops.OnesLike()

        # adaptive float16, it will be nan when used -1e9
        self.value = -1e10

    def construct(self, q, k, v, mask=None):
        """
        :param q: (L,  N, E) L is the query sequence length, N is the batch size, E is the embedding dimension
        :param k: (L', N, E)
        :param v: (L', N, E)
        :param mask: (N, L')
        :return: (L, N, E)
        """
        l, bs, _ = q.shape
        l_, _, _ = k.shape

        # (L,N,E) => (L,N,H,D) H is the head nums, D is the dim of each block
        q = self.reshape(self.q_dense(q), (l, bs, self.h, self.d_k))
        k = self.reshape(self.k_dense(k), (l_, bs, self.h, self.d_k))
        v = self.reshape(self.v_dense(v), (l_, bs, self.h, self.d_k))

        q = self.transpose(q, (1, 2, 0, 3))  # (L, N,H,D) => (N,H,L, D)
        v = self.transpose(v, (1, 2, 0, 3))  # (L',N,H,D) => (N,H,L',D)
        k = self.transpose(k, (1, 2, 3, 0))  # (L',N,H,D) => (N,H,D, L')

        # (N,H,L,D) x (N,H,D,L') => (N,H,L,L')
        score = self.batch_mul(q, k) / self.sqrt(self.cast(self.d_k, q.dtype))

        if self.has_mask:
            # (N,L') => (N,1,1,L')
            score = self.cast(score + mask[:, None, None, :] * self.value, q.dtype)

        score = self.softmax(score)
        score = self.dropout(score)

        # (N,H,L,L') x (N,H,L',D) => (N,H,L,D)
        score = self.batch_mul(score, v)
        # (N,H,L,D) => (L,N,H,D) => (L,N,E)
        output = self.transpose(score, (2, 0, 1, 3))
        output = self.reshape(output, (l, bs, -1))
        output = self.out_proj(output)
        return output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()


class TransformerEncoderLayer(nn.Cell):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # self.self.attn = nn.MultiHeadAttention(batch_size=8,src_seq_length=40,tgt_seq_length=40,hidden_size=d_model,num_heads=nhead,compute_dtype=ms.float32)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward, weight_init=KaimingUniform(), bias_init=UniformBias([dim_feedforward, d_model]))
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model, weight_init=KaimingUniform(), bias_init=UniformBias([d_model, dim_feedforward]))

        self.norm1 = nn.LayerNorm((d_model, ))
        self.norm2 = nn.LayerNorm((d_model, ))
        self.dropout1 = nn.Dropout(keep_prob=1 - dropout)
        self.dropout2 = nn.Dropout(keep_prob=1 - dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward_post(self, src, mask, pos):
        q = k = src + pos
        # attention + dropout
        src2 = self.self_attn(q, k, src, mask)
        src2 = self.dropout1(src2)

        src = src + src2
        # layer normal
        src = self.norm1(src)
        # linear + relu + dropout + linear + dropout
        src2 = self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src)))))

        src = src + src2
        # layer normal
        src = self.norm2(src)
        return src

    def forward_pre(self, src, mask, pos):
        src2 = self.norm1(src)
        q = k = src2 + pos
        src2 = self.dropout1(self.self_attn(q, k, src, mask))

        src = src + src2
        src2 = self.norm2(src)
        src2 = self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src2)))))

        src = src + src2
        return src

    def construct(self, src, mask, pos):
        if self.normalize_before:
            return self.forward_pre(src, mask, pos)
        return self.forward_post(src, mask, pos)


class TransformerEncoderLayerWithHook(nn.Cell):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = nn.MultiHeadAttention(batch_size=8,src_seq_length=40,tgt_seq_length=40,hidden_size=d_model,num_heads=nhead,compute_dtype=ms.float32)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward, weight_init=initializer(XavierUniform(), (dim_feedforward, d_model)), bias_init=UniformBias([dim_feedforward, d_model]))
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model, weight_init=initializer(XavierUniform(), (d_model, dim_feedforward)), bias_init=UniformBias([d_model, dim_feedforward]))

        self.norm1 = nn.LayerNorm((d_model, ))
        self.norm2 = nn.LayerNorm((d_model, ))
        self.dropout1 = nn.Dropout(keep_prob=1 - dropout)
        self.dropout2 = nn.Dropout(keep_prob=1 - dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward_post(self, src, mask, pos, attn_ffn_selection, index):
        q = k = src + pos
        # attention + dropout
        src2 = self.dropout1(self.self_attn(q, k, src, mask))
        # ------------hook---------------
        mix_gates = attn_ffn_selection[index][2].astype(src2.dtype)
        mix_gates = mix_gates.view(1, -1, 1)
        src2 = src2 * mix_gates
        # --------------------------------

        src = src + src2
        # layer normal
        src = self.norm1(src)
        # linear + relu + dropout + linear + dropout
        src2 = self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src)))))
        # hook
        mix_gates = attn_ffn_selection[index][-1].astype(src2.dtype)
        mix_gates = mix_gates.view(1, -1, 1)
        src2 = src2 * mix_gates
        index += 1

        src = src + src2
        # layer normal
        src = self.norm2(src)
        return src, index

    def forward_pre(self, src, mask, pos, attn_ffn_selection, index):
        src2 = self.norm1(src)
        q = k = src2 + pos
        src2 = self.dropout1(self.self_attn(q, k, src, mask))

        # ------------hook---------------
        mix_gates = attn_ffn_selection[index][:3]
        mix_gates = mix_gates.view(1, -1, 1)
        src2 = src2 * mix_gates
        # --------------------------------

        src = src + src2
        src2 = self.norm2(src)
        src2 = self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src2)))))
        # hook
        mix_gates = attn_ffn_selection[index][-3:]
        mix_gates = mix_gates.view(1, -1, 1)
        src2 = src2 * mix_gates
        index += 1

        src = src + src2
        return src, index

    def construct(self, src, mask, pos, attn_ffn_selection, index):
        if self.normalize_before:
            return self.forward_pre(src, mask, pos, attn_ffn_selection, index)
        return self.forward_post(src, mask, pos, attn_ffn_selection, index)


class TransformerEncoder(nn.Cell):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, num_layers, normalize_before=None, use_selector=True):
        super().__init__()
        self.use_selector = use_selector
        self.layers = nn.CellList()
        if use_selector:
            for _ in range(num_layers):
                layer = TransformerEncoderLayerWithHook(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, normalize_before=normalize_before)
                self.layers.append(layer)
        else:
            for _ in range(num_layers):
                layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, normalize_before=normalize_before)
                self.layers.append(layer)

        self.num_layers = num_layers
        self.norm = nn.LayerNorm((d_model, )) if normalize_before else None

    def construct(self, src, mask, pos, attn_ffn_selection=None, index=0):
        output = src
        if self.use_selector:
            for layer in self.layers:
                output, index = layer(output, mask=mask, pos=pos, attn_ffn_selection=attn_ffn_selection, index=index)
        else:
            for layer in self.layers:
                output = layer(output, mask=mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Cell):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, has_mask=False)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward, weight_init=KaimingUniform(), bias_init=UniformBias([dim_feedforward, d_model]))
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model, weight_init=KaimingUniform(), bias_init=UniformBias([d_model, dim_feedforward]))

        self.norm1 = nn.LayerNorm((d_model, ))
        self.norm2 = nn.LayerNorm((d_model, ))
        self.norm3 = nn.LayerNorm((d_model, ))
        self.dropout1 = nn.Dropout(keep_prob=1 - dropout)
        self.dropout2 = nn.Dropout(keep_prob=1 - dropout)
        self.dropout3 = nn.Dropout(keep_prob=1 - dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.gpu_flag = True if context.get_context("device_target") == "GPU" else False

    def forward_post(self, tgt, memory, mask, pos, query_pos):
        q = k = tgt + query_pos
        # attention + dropout
        tgt2 = self.dropout1(self.self_attn(q, k, tgt))
        tgt = tgt + tgt2

        # this is a hack
        if self.gpu_flag:
            tgt = ops.Cast()(tgt, ms.float16)

        # layer normal
        tgt = self.norm1(tgt)
        # attention + dropout
        tgt2 = self.dropout2(self.multihead_attn(q=tgt + query_pos, k=memory + pos, v=memory, mask=mask))
        tgt = tgt + tgt2
        # layer normal
        tgt = self.norm2(tgt)
        # linear + relu + dropout + linear + dropout
        tgt2 = self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt)))))
        tgt = tgt + tgt2
        # layer normal
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, mask, pos, query_pos):
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos
        tgt2 = self.dropout1(self.self_attn(q, k, tgt))

        tgt = tgt + tgt2

        # this is a hack
        if self.gpu_flag:
            tgt = ops.Cast()(tgt, ms.float16)

        tgt2 = self.norm2(tgt)
        tgt2 = self.dropout2(self.multihead_attn(q=tgt2 + query_pos, k=memory + pos, v=memory, mask=mask))
        tgt = tgt + tgt2
        tgt2 = self.norm3(tgt)
        tgt2 = self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt2)))))
        tgt = tgt + tgt2
        return tgt

    def construct(self, tgt, memory, mask, pos, query_pos):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, mask, pos, query_pos)
        return self.forward_post(tgt, memory, mask, pos, query_pos)


class TransformerDecoder(nn.Cell):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, num_layers, normalize_before, return_intermediate=False):
        super().__init__()

        self.layers = nn.CellList()
        for _ in range(num_layers):
            layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
            self.layers.append(layer)

        self.num_layers = num_layers
        self.norm = nn.LayerNorm((d_model, ))
        self.return_intermediate = return_intermediate

        self.stack = ops.Stack()
        self.expand_dims = ops.ExpandDims()

    def construct(self, tgt, memory, mask, pos, query_pos):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory=memory, mask=mask, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate[-1] = output

        if self.return_intermediate:
            return self.stack(intermediate)

        return self.expand_dims(output, 0)


class Transformer(nn.Cell):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False, return_intermediate_dec=False):
        super().__init__()

        self.encoder = TransformerEncoder(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, num_layers=num_encoder_layers)

        self.decoder = TransformerDecoder(d_model=d_model,
                                          nhead=nhead,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation=activation,
                                          normalize_before=normalize_before,
                                          num_layers=num_decoder_layers,
                                          return_intermediate=return_intermediate_dec)

        self.d_model = d_model
        self.nhead = nhead
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.zero_like = ops.ZerosLike()
        self.tile = ops.Tile()
        self.expand_dims = ops.ExpandDims()

    def construct(self, src, mask, query_embed, pos_embed):
        # (N,C,H,W) to (H*W,N,C)
        bs, c, h, w = src.shape
        src = self.reshape(src, (bs, c, h * w))
        src = self.transpose(src, (2, 0, 1))
        pos_embed = self.reshape(pos_embed, (bs, c, h * w))
        pos_embed = self.transpose(pos_embed, (2, 0, 1))

        # (N,H,W) to (N,H*W)
        mask = self.reshape(mask, (bs, h * w))

        # (queries, hidden_dim) => (queries, N, hidden_dim)
        query_embed = self.expand_dims(query_embed, 1)
        query_embed = self.tile(query_embed, (1, bs, 1))

        tgt = self.zero_like(query_embed)

        memory = self.encoder(src=src, mask=mask, pos=pos_embed)
        hs = self.decoder(tgt=tgt, memory=memory, mask=mask, pos=pos_embed, query_pos=query_embed)
        return self.transpose(hs, (0, 2, 1, 3))


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )
