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
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def main():
    d_model = 512
    vocab = 1000

    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

    emb = Embeddings(d_model, vocab)
    embr = emb(x)
    # print(embr)
    # print(embr.shape)

    dropout = 0.1
    max_len = 60

    x = embr
    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(x)

    # size = 5
    # sm = subsequent_mask(size)
    # print(sm)

    # x = Variable(torch.randn(5, 5))
    # print(x)
    #
    # mask = Variable(torch.zeros(5, 5))
    # print(mask)
    #
    # y = x.masked_fill(mask == 0, -1e9)
    # print(y)

    query = key = value = pe_result
    attn, p_attn = attention(query, key, value)
    # print(attn)
    # print(attn.shape)
    # print(p_attn)

    head = 8
    embedding_dim = 512
    dropout = 0.2
    mask = Variable(torch.zeros(8, 4, 4))
    mha = MultiHeadedAttention(head, embedding_dim, dropout)
    mha_result = mha(query, key, value, mask)
    print(mha_result)
    pass


if __name__ == '__main__':
    main()
