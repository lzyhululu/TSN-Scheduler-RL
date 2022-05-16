"""transformer method needed in tsn scheduler"""
# 预定义的网络层torch.nn
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import copy

# torch中变量封装函数Variable
from torch.autograd import Variable


# Embeddings 文本嵌入层，有两个一模一样的嵌入层，共享参数
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """ d_model: 词嵌入的维度, vocab: 词表的大小"""
        super(Embeddings, self).__init__()
        # lut: 词嵌入对象， 利用nn中的预定义层Embedding
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """该层的前向传播逻辑，当传给该类实例化对象参数的时候，自动调用该函数
        参数x: Embedding层是首层，所以代表输入给模型的文本通过词汇映射后的张量
        """
        return self.lut(x) * math.sqrt(self.d_model)


# Positional Encoding, 位置编码器
class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        """
        位置编码器类的初始化函数

        共有三个参数，分别是
        d_model：词嵌入维度
        dropout: dropout触发比率
        max_len：每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings
        # 初始化位置编码矩阵， 0阵，大小max_len X d_model
        pe = torch.zeros(max_len, d_model)
        # 初始化一个绝对位置矩阵，即用索引去表示，扩展为max_len X 1
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 对输入的大小进行适配
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(1 - mask)


# Attention
def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    # 首先取query的最后一维大小，一般情况下等同于词嵌入维度
    d_k = query.size(-1)
    # 按照注意力计算公式，query与key转置相乘 除以一个缩放系数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 判断是否使用，mask = 0 的位置用很小的数值填充，不可能被选中
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # softmax 得到注意力张量
    p_attn = F.softmax(scores, dim=-1)
    # 注意dropout是对象
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """Produce N identical layers."""
    # 对module进行N次深度拷贝
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % head == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // head
        self.head = head
        # Q, K, V各自需要一个线性层，最后合并还需要一个，同时内部变换矩阵必定为方阵，因为不改变形状
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(0)
        # 代表有多少样本
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # 首先利用zip，将输入的Q，K，V和三个线性层组合到一起
        # view对线性变换结果进行维度重塑，第二维度自适应
        # 转置是为了让句子长度维度和词向量维度能够相邻
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # contiguous让转置后的张量能够执行转置
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.head * self.d_k)
        # 用第四个线性层返回
        return self.linears[-1](x)


# 前馈全连接层，防止注意力机制对复杂过程的拟合程度不够
class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: 来自上一层的输出
        return self.w2(self.dropout(F.relu(self.w1(x))))


# 规范化层
class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        """eps 很小，防止出现分母除0的那种错误"""
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # *为同型点乘
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SublayerConnection(nn.Module):
    """实现子层连接结构的类"""

    def __init__(self, size, dropout):
        """size为词嵌入维度的大小"""
        super(SublayerConnection, self).__init__()
        # 实例化一个规范化层对象
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        # 接收上一个层或子层的输入作为第一个参数, 该子层链接中的子层函数作为第二个参数
        # sublayer: 代表该子层连接中子层函数
        # 首先将x规范化，送入子层函数处理，结果进入dropout层，最后进行残差连接
        return x + self.dropout(sublayer(self.norm(x)))


# 编码器层
class EncoderLayer(nn.Module):
    """EncoderLayer is made up of two sublayer: self-attn and feed forward"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        """size，词嵌入维度大小，self_attn， 多头 自 注意力机制实例化对象"""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 需要两个子层连接结构
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # attention sub layer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # feed forward sub layer
        return self.sublayer[1](x, self.feed_forward)


# 编码器的作用：对输入特征进行指定的特征提取过程，N个编码器层堆叠而成
class Encoder(nn.Module):
    """
    Encoder
    The encoder is composed of a stack of N=6 identical layers.
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # 此规范化层用在编码器的最后面
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# 每个解码器层根据给定的输入向目标方向进行特征提取操作
class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """size, 词嵌入维度的大小，也代表解码器层的尺寸。
        self_attn，多头自注意力对象
        src_attn，多头注意力对象 Q！= k = v
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 克隆三个子层连接对象
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        # x，上一层输入，memory代表编码器层的语义存储变量，即编码器的输出
        # 源数据掩码张量，目标数据掩码张量
        m = memory
        # 解码器在生成第一个字符或词汇时，其实已经传入了第一个字符以便计算损失
        # 但我们不希望在生成第一个字符的时候模型能够利用这一个信息，因此将其遮掩
        # 生成字符时只能使用之前的信息
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 使用常规注意力机制，q为输入x；k，v是编码层输出memory，此处遮掩主要是为了遮蔽掉对结果没有意义的值
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 前馈全连接子层
        return self.sublayer[2](x, self.feed_forward)


# 解码器会根据编码器的结果以及上一次预测的结果，对下一次可能出现的‘值’进行特征表示
class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        # N，解码器层个数N
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        # d_model, 词嵌入维度，vocab， 词表大小，结果在词表中选择
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# Model Architecture, 编码器，解码器结构
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # input embedding module(input embedding + positional encode)
        self.tgt_embed = tgt_embed  # ouput embedding module
        self.generator = generator  # output generation module

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        memory = self.encode(src, src_mask)
        res = self.decode(memory, src_mask, tgt, tgt_mask)
        return res

    def encode(self, src, src_mask):
        src_embeds = self.src_embed(src)
        return self.encoder(src_embeds, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        target_embeds = self.tgt_embed(tgt)
        return self.decoder(target_embeds, memory, src_mask, tgt_mask)


# Full Model, 构建完整的用于训练的模型
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    构建模型
    params:
        src_vocab:
        tgt_vocab:
        N: 编码器和解码器堆叠基础模块的个数
        d_model: 模型中embedding的size，默认512
        d_ff: FeedForward Layer层中embedding的size，默认2048
        h: MultiHeadAttention中多头的个数，必须被d_model整除
        dropout:
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class ScheT(nn.Module):
    def __init__(self, *, flow_nums, patch_size, slot_nums, dim,
                 depth, heads, mlp_dim, pool='cls', dim_head, ):
        super().__init__()
        # 一条流量特征包含的信息数，


def main():
    s = ScheT(
        flow_nums=120,
        patch_size=16,
        slot_nums=512,
        dim=1024,
        depth=6,
        heads=6,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )


if __name__ == '__main__':
    main()
