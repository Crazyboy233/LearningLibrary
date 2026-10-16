"""
Transformer（Attention Is All You Need）从零实现 + 训练示例 —— PyTorch 版

和 `DeepLearningFramework/tf/work/02_train.py` 是同一个例子的两次实现：
常量、数据生成、网络结构、打印格式、注意力可视化全部保持一致，
所以两个文件可以直接左右对照，看同一件事在两个框架里怎么写。

任务：合成序列任务（默认“序列反转”，可选“原样复制”）
      输入  [3 7 1 5] EOS   ->  输出  SOS [5 1 7 3] EOS

本文件与论文的对应关系（括号内为论文小节号）：
    1. 配置                      —— 表 3 base 列（这里按本地规模缩小）
    2. 缩放点积注意力 + 多头      —— 3.2.1 / 3.2.2 式(1)
    3. 正弦位置编码               —— 3.5
    4. 残差 + LayerNorm           —— 3.1
    5. 逐位置前馈网络 FFN          —— 3.3 式(2)
    6. 三种掩码                   —— 3.2.3（padding mask + look-ahead mask）
    7. 嵌入/输出投影权重共享       —— 3.4
    8. Dropout                    —— 5.4
    9. Adam + warmup 学习率        —— 5.3 式(3)
   10. 标签平滑                    —— 5.4
   11. 自回归贪心解码             —— 3.1 + 6.1（论文用 beam search=4，这里用贪心便于阅读）

┌────────────────────────┬──────────────────────────────────┬──────────────────────────────────┐
│ 要做的事                │ TensorFlow 写法                   │ PyTorch 写法                      │
├────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ 模型基类                │ tf.keras.Model + call()          │ nn.Module + forward()             │
│ 线性层                  │ tf.keras.layers.Dense            │ nn.Linear                         │
│ 层归一化                │ tf.keras.layers.LayerNormalization│ nn.LayerNorm                     │
│ 字典（词嵌入）           │ tf.keras.layers.Embedding        │ nn.Embedding                      │
│ 堆叠 N 层               │ python list + for 循环           │ nn.ModuleList（必须用它，否则参数注册不上）│
│ 张量形状                │ (B, T, C) 全程一致                │ (B, T, C)；注意 nn.MultiheadAttention 是 (T,B,C)│
│ 训练开关                │ 每个层手动传 training=True/False  │ model.train() / model.eval() 全局切换│
│ 自动求导                │ with tf.GradientTape() as tape:  │ loss.backward()                   │
│ 反向传播                │ tape.gradient(loss, vars)        │ 梯度直接写进 tensor.grad            │
│ 参数更新                │ optimizer.apply_gradients(...)   │ optimizer.step()                  │
│ 梯度清零                │ 不需要（tape 是局部的）            │ optimizer.zero_grad()（默认会累加） │
│ 梯度裁剪                │ tf.clip_by_global_norm           │ torch.nn.utils.clip_grad_norm_    │
│ 损失函数                │ categorical_crossentropy + 手动 mask│ nn.CrossEntropyLoss(ignore_index=PAD)│
│ 学习率调度              │ 继承 LearningRateSchedule         │ LambdaLR（传一个 lambda 就行）      │
│ 权重共享                │ 复用 .embeddings 变量             │ out_proj.weight = embedding.weight│
│ 自定义层里的掩码参数名    │ 不能叫 mask（见文末注释）          │ 随便叫，没有保留字限制              │
└────────────────────────┴──────────────────────────────────┴──────────────────────────────────┘

运行（先激活虚拟环境）：
    cd DeepLearningFramework/pytorch
    source torch-env/bin/activate
    python work/02_train.py
"""
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 可选：装了 matplotlib 就顺手把注意力矩阵画成热力图
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# ============================ 1. 配置 ============================
PAD, SOS, EOS = 0, 1, 2          # 三个特殊 token：填充 / 起始 / 结束
N_SYMBOLS = 10                   # “内容”符号个数（0~9，对应 id 3~12）
VOCAB_SIZE = N_SYMBOLS + 3       # 13

MAX_SYMBOLS = 8                  # 一条序列最多 8 个内容符号
SEQ_LEN = MAX_SYMBOLS + 1        # 统一补齐到 9；形状固定，训练更稳定也更快

# 论文 base 模型用 D_MODEL=512 / D_FF=2048 / N_HEADS=8 / N_LAYERS=6，
# 这里为本地演示缩小，重点是把结构讲清楚。
D_MODEL = 64
D_FF = 256                       # 论文取 4 * D_MODEL
N_HEADS = 4                      # 每个头的维度 d_k = d_v = 64 / 4 = 16
N_LAYERS = 2
DROPOUT = 0.1                    # 论文 5.4：P_drop = 0.1

BATCH_SIZE = 64
TRAIN_STEPS = 2000
WARMUP_STEPS = 400               # 论文 5.3：warmup_steps = 4000
TASK = "reverse"                 # "reverse"（反转）| "copy"（复制）
SEED = 42

# 本机是 Apple Silicon，MPS 可用；想跑 GPU 改成 "mps"（或 "cuda"）即可。
# 默认用 CPU，是为了跟 TensorFlow 那版（tf-env 只有 CPU）在同等条件下对比速度。
DEVICE = "cpu"

RESULTS_DIR = "./results_transformer"


# ============================ 2. 数据 ============================
rng = np.random.default_rng(SEED)


def random_batch(batch_size, task=TASK):
    """在线生成一个 batch：长度随机，便于检验模型的泛化能力。

    返回：
        src     (B, SEQ_LEN)  源序列，[符号...] + EOS，其余位补 PAD
        tgt_in  (B, SEQ_LEN)  解码器输入，SOS + [目标符号...]
        tgt_out (B, SEQ_LEN)  监督标签，  [目标符号...] + EOS
        lens    (B,)          每个样本的真实长度（不含 EOS）
    """
    lens = rng.integers(1, MAX_SYMBOLS + 1, size=batch_size)
    src = np.full((batch_size, SEQ_LEN), PAD, np.int32)
    tgt_in = np.full((batch_size, SEQ_LEN), PAD, np.int32)
    tgt_out = np.full((batch_size, SEQ_LEN), PAD, np.int32)

    for i, L in enumerate(lens):
        s = rng.integers(0, N_SYMBOLS, size=L) + 3          # 符号 id 从 3 开始
        t = s[::-1] if task == "reverse" else s
        src[i, :L + 1] = np.append(s, EOS)                  # [s... ] EOS
        tgt_in[i, :L + 1] = np.append(SOS, t)               # SOS [t... ]
        tgt_out[i, :L + 1] = np.append(t, EOS)              # [t... ] EOS

    # PyTorch 的张量要显式放到目标设备上（TF 是自动跟着计算图走的）
    to = lambda a: torch.as_tensor(a, dtype=torch.long, device=DEVICE)
    return to(src), to(tgt_in), to(tgt_out), lens


def sym_name(idx):
    """把 token id 打印成人能看懂的样子。"""
    idx = int(idx)
    if idx == PAD:
        return "_"
    if idx == SOS:
        return "<s>"
    if idx == EOS:
        return "</s>"
    return str(idx - 3)


def seq_str(ids, stop_at_pad=True):
    out = []
    for i in ids:
        if stop_at_pad and i == PAD:
            break
        out.append(sym_name(i))
    return " ".join(out)


def seq_tokens(ids):
    """取出有效 token 列表：遇到 EOS / PAD 就截断（不含 EOS 本身），用于比对。"""
    out = []
    for i in ids:
        i = int(i)
        if i == PAD or i == EOS:
            break
        out.append(sym_name(i))
    return out


# ============================ 3. 掩码 ============================
NEG_INF = -1e9


def padding_mask(seq):
    """padding 掩码：形状 (B, 1, 1, T)，PAD 位置为 -1e9，其余为 0（加性掩码）。

    PyTorch 里 mask 只是一个普通张量，参与加法即可；不像 TF 那样要小心
    不要跟框架内置的掩码机制撞名。
    """
    keep = (seq != PAD).float()
    return (1.0 - keep)[:, None, None, :] * NEG_INF


def causal_mask(size, device):
    """look-ahead 掩码（论文 3.2.3）：下三角为 0、上三角为 -1e9。

    torch.tril 取矩阵的下三角，比 TF 的 tf.linalg.band_part 直白一些。
    位置 i 只能看到 <= i 的位置，从而保证解码器的自回归性质。
    """
    tri = torch.tril(torch.ones(size, size, device=device))
    return (1.0 - tri)[None, None, :, :] * NEG_INF


# ============================ 4. 缩放点积注意力 + 多头 ============================
class MultiHeadAttention(nn.Module):
    """MultiHead(Q,K,V) = Concat(head_1..head_h) W^O
       head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
       Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads          # 论文 3.2.2：d_k = d_v = d_model / h
        self.d_v = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)  # W^O
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """(B, T, d_model) -> (B, n_heads, T, d_k)

        view + transpose 是 PyTorch 的惯用手法；TF 那边是 reshape + tf.transpose。
        注意 transpose 后内存不连续，后面要再 transpose 回来时必须 .contiguous()。
        """
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, q, k, v, attn_mask=None):
        B, Tq, _ = q.shape

        q = self._split_heads(self.wq(q))       # (B, h, Tq, d_k)
        k = self._split_heads(self.wk(k))       # (B, h, Tk, d_k)
        v = self._split_heads(self.wv(v))       # (B, h, Tk, d_v)

        # 式(1)：先点积、再除以 sqrt(d_k) —— d_k 越大点积数值越大，
        # 不缩放会让 softmax 落入梯度极小的饱和区（论文 3.2.1 脚注 4）。
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)   # (B, h, Tq, Tk)

        if attn_mask is not None:
            scores = scores + attn_mask         # 掩掉的位置 softmax 后权重 ~ 0

        attn = F.softmax(scores, dim=-1)        # 每行归一化成权重分布
        attn = self.attn_dropout(attn)

        out = attn @ v                          # (B, h, Tq, d_v) 值的加权和
        out = out.transpose(1, 2).contiguous().reshape(B, Tq, self.d_model)
        return self.wo(out), attn               # 拼接后过一次线性投影


# ============================ 5. 位置编码 ============================
class PositionalEncoding(nn.Module):
    """论文 3.5：PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(...)

    用固定公式而不是可学习参数，好处是可以外推到训练时没见过的更长的序列（表 3 (E)）。
    这里在 forward 里现算，所以 batch 多长都能处理。
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        T = x.size(1)
        pos = torch.arange(T, dtype=torch.float32, device=x.device)[:, None]        # (T, 1)
        i = torch.arange(self.d_model, dtype=torch.float32, device=x.device)[None, :]  # (1, d)
        angle = pos / torch.pow(10000.0, (2 * (i // 2)) / float(self.d_model))
        even = (i % 2 == 0)                                                        # 偶数列 sin、奇数列 cos
        pe = torch.where(even, torch.sin(angle), torch.cos(angle))                 # (T, d)
        return x + pe[None, :, :]


# ============================ 6. 前馈网络 / 编码器层 / 解码器层 ============================
def pointwise_ffn(d_model, d_ff, dropout):
    """式(2)：FFN(x) = ReLU(x W1 + b1) W2 + b2，对每个位置独立且共享参数。

    TF 用 tf.keras.Sequential，PyTorch 用 nn.Sequential —— 这一步几乎是同构的。
    """
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, d_model),
        nn.Dropout(dropout),
    )


class EncoderLayer(nn.Module):
    """论文 3.1 编码器层：自注意力 + FFN，两个子层都是 LayerNorm(x + Sublayer(x))。"""

    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = pointwise_ffn(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # 不用手动传 training 标志：nn.Dropout 自己看 model.train()/model.eval()
        h, attn = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.ln1(x + self.drop1(h))            # 残差 + LayerNorm
        h = self.ffn(x)
        x = self.ln2(x + self.drop2(h))
        return x, attn


class DecoderLayer(nn.Module):
    """论文 3.1 解码器层：比编码器多一个「编码器-解码器注意力」子层（共 3 个子层）。"""

    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)   # 带 look-ahead 掩码
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)  # Q 来自解码器，K/V 来自编码器
        self.ffn = pointwise_ffn(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln3 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        h, self_attn = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.ln1(x + self.drop1(h))
        h, cross_attn = self.cross_attn(x, enc_out, enc_out, attn_mask=src_mask)
        x = self.ln2(x + self.drop2(h))
        h = self.ffn(x)
        x = self.ln3(x + self.drop3(h))
        return x, self_attn, cross_attn


# ============================ 7. Transformer ============================
class Transformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, d_ff=D_FF,
                 n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model

        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # 必须用 nn.ModuleList：普通 python list 里的子模块不会被注册，
        # 参数不会进 model.parameters()，优化器就更新不到它们（这是新手最常踩的坑）。
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

        # 3.4：输出投影与输入嵌入共享权重。
        # nn.Embedding.weight 形状是 (vocab, d_model)，nn.Linear.weight 是 (out, in)，
        # 正好都是 (vocab, d_model)，所以可以直接指向同一个 Parameter；
        # 两条路径的梯度会自动累加到同一份参数上，不需要额外处理。
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.encoder_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """按 Keras 的默认初始化来设参数，保证两版结果可比。

        这是「把 TF 代码逐行翻译成 PyTorch」时最容易被忽略的坑——两边的默认初始化完全不同：
          - nn.Embedding 默认 N(0, 1)；Keras Embedding 默认 U(-0.05, 0.05)，差了 20 倍。
            而嵌入向量会先乘 sqrt(d_model)=8，同一份权重又要拿去做输出投影，
            权重一大，logits 直接顶进 softmax 饱和区，梯度极小、收敛极慢。
          - nn.Linear 默认 U(±1/sqrt(fan_in))；Keras Dense 默认 glorot_uniform
            U(±sqrt(6/(fan_in+fan_out)))，两者差约 1.7 倍。
        nn.init.xavier_uniform_ 就是 PyTorch 版的 glorot_uniform，正好一一对应。

        实测：不统一初始化，同样超参、同样 2000 步，准确率只有 0.64；统一后到 0.99。
        """
        for name, m in self.named_modules():
            if name == "out_proj":
                continue        # 它的 weight 和 encoder_embedding 是同一个 Parameter，别重复初始化
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.05, 0.05)

    def encode(self, src, src_mask):
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)   # 3.4：乘 sqrt(d_model)
        x = self.emb_dropout(self.pos_encoding(x))
        attns = []
        for layer in self.enc_layers:
            x, a = layer(x, src_mask)
            attns.append(a)
        return x, attns

    def decode(self, tgt_in, enc_out, src_mask, tgt_mask):
        x = self.decoder_embedding(tgt_in) * math.sqrt(self.d_model)
        x = self.emb_dropout(self.pos_encoding(x))
        self_attns, cross_attns = [], []
        for layer in self.dec_layers:
            x, sa, ca = layer(x, enc_out, src_mask, tgt_mask)
            self_attns.append(sa)
            cross_attns.append(ca)
        return x, self_attns, cross_attns

    def output_projection(self, x):
        return self.out_proj(x)

    def forward(self, src, tgt_in, return_attention=False):
        src_mask = padding_mask(src)                                              # 屏蔽输入 PAD
        tgt_mask = padding_mask(tgt_in) + causal_mask(tgt_in.size(1), tgt_in.device)  # PAD + 未来位置

        enc_out, enc_attns = self.encode(src, src_mask)
        dec_out, dec_self_attns, dec_cross_attns = self.decode(tgt_in, enc_out, src_mask, tgt_mask)
        logits = self.output_projection(dec_out)

        if return_attention:
            return logits, {"enc_self": enc_attns, "dec_self": dec_self_attns, "dec_cross": dec_cross_attns}
        return logits


# ============================ 8. 损失与学习率 ============================
def masked_loss(logits, labels):
    """标签平滑（5.4）+ 只在真实 token 上求平均（忽略 PAD 位置）。

    PyTorch 一行就够：ignore_index=PAD 且 reduction='mean'，
    正好等价于 TF 版里「手动乘掩码再除以掩码之和」的写法。
    label_smoothing=0.1 会让模型学到“不那么自信”的分布，
    困惑度（PPL）变差但翻译准确率/BLEU 反而提升。
    """
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=PAD)
    return loss_fn(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))


class TransformerLRSchedule:
    """论文 5.3 式(3)：lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)

    前 warmup_steps 步线性升温，之后按步数平方根倒数衰减。
    这种“先热身再退火”对 Post-LN 的 Transformer 尤其关键。

    TF 那边要继承 LearningRateSchedule 写个类；PyTorch 直接丢给 LambdaLR 当 lambda 用，
    配一个 lr=1.0 的基础学习率，乘出来的就是绝对学习率。
    """

    def __init__(self, d_model, warmup_steps):
        self.d_model = float(d_model)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = max(float(step), 1.0)
        return self.d_model ** -0.5 * min(step ** -0.5, step * self.warmup_steps ** -1.5)


# ============================ 9. 自回归贪心解码 ============================
@torch.no_grad()
def greedy_decode(model, src, max_tokens=SEQ_LEN):
    """逐 token 生成：每步只把已生成的部分喂给解码器，取概率最大的下一个 token。"""
    model.eval()                                   # 关掉 dropout
    src_mask = padding_mask(src)
    enc_out, _ = model.encode(src, src_mask)

    B = src.size(0)
    ys = torch.full((B, 1), SOS, dtype=torch.long, device=src.device)
    for _ in range(max_tokens):
        tgt_mask = causal_mask(ys.size(1), ys.device)                 # 生成时没有 PAD
        dec_out, _, _ = model.decode(ys, enc_out, src_mask, tgt_mask)
        logits = model.output_projection(dec_out)[:, -1, :]           # 只关心最后一个位置
        nxt = logits.argmax(dim=-1)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
    return ys[:, 1:].cpu().numpy()                                    # 去掉开头那个 SOS


def evaluate(model, n_batches=6):
    """指标：整条序列完全正确才算对（exact match）。"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            src, _, tgt_out, _ = random_batch(BATCH_SIZE, TASK)
            pred = greedy_decode(model, src)
            tgt_np = tgt_out.cpu().numpy()
            for i in range(src.size(0)):
                correct += int(seq_tokens(tgt_np[i]) == seq_tokens(pred[i]))
                total += 1
    return correct / total


# ============================ 10. 注意力可视化 ============================
RAMP = " .:-=+*#%@"


def _sym_short(idx):
    return {PAD: "_", SOS: "s", EOS: "E"}.get(int(idx), str(int(idx) - 3))


def _print_matrix(mat, row_labels, col_labels, title):
    print(f"  {title}   (行=query，列=key)")
    print("         " + "".join(f"{c:>2}" for c in col_labels))
    for i, lab in enumerate(row_labels):
        row = "".join(RAMP[min(int(v * 9), 9)] for v in mat[i, :len(col_labels)])
        print(f"    {lab:>3} |{row}|")


def show_attention(model, src=None):
    """打印最后一层的注意力矩阵。

    编码器自注意力：源序列内部的位置关系。
    解码器跨注意力：查询来自解码器、键值来自编码器 —— 反转任务的对齐模式
                    （哪个输出位置该看哪个输入位置）在这里看得最清楚。
    """
    model.eval()
    if src is None:                      # 固定一条样本，方便观察对齐模式
        s = np.array([2, 7, 0, 5, 9, 3], np.int32) + 3
        src_np = np.full((1, SEQ_LEN), PAD, np.int32)
        src_np[0, :len(s) + 1] = np.append(s, EOS)
    else:
        src_np = src.cpu().numpy()

    L = int(np.sum(src_np[0] != PAD)) - 1                    # 真实符号数（去掉 EOS）
    rev = (src_np[0][:L] - 3)                                # 目标符号 = 源符号倒序
    tgt_np = np.full((1, SEQ_LEN), PAD, np.int32)
    tgt_np[0, :L + 1] = np.append(SOS, rev[::-1] + 3)         # 解码器输入：SOS + 倒序目标

    with torch.no_grad():
        src_t = torch.as_tensor(src_np, dtype=torch.long, device=DEVICE)
        tgt_t = torch.as_tensor(tgt_np, dtype=torch.long, device=DEVICE)
        _, info = model(src_t, tgt_t, return_attention=True)

    enc_self = info["enc_self"][-1][0].cpu().numpy()          # (h, T, T)
    dec_cross = info["dec_cross"][-1][0].cpu().numpy()        # (h, Tq, Tk)

    src_lab = [_sym_short(v) for v in src_np[0][:L + 1]]
    tgt_lab = [_sym_short(v) for v in tgt_np[0][:L + 1]]

    print("\n  ===== 注意力可视化（第 0 个样本）=====")
    print(f"  源序列: {seq_str(src_np[0])}")
    print(f"  目标序列: {seq_str(np.append(np.append(SOS, rev[::-1] + 3), EOS))}")
    for h in range(min(2, enc_self.shape[0])):
        _print_matrix(enc_self[h, :L + 1, :L + 1], src_lab, src_lab,
                      f"[编码器自注意力 head {h}]")
    for h in range(min(2, dec_cross.shape[0])):
        _print_matrix(dec_cross[h, :L + 1, :L + 1], tgt_lab, src_lab,
                      f"[解码器跨注意力 head {h}]")
    print("  （跨注意力的亮斑应落在反对角线：输出第 j 位看输入第 L-1-j 位）")

    if HAS_PLT:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        h = min(2, enc_self.shape[0])
        fig, axes = plt.subplots(1, 2 * h, figsize=(3 * 2 * h, 3.2))
        for k in range(h):
            axes[k].imshow(enc_self[k, :L + 1, :L + 1], cmap="viridis")
            axes[k].set_title(f"enc self h{k}")
            axes[h + k].imshow(dec_cross[k, :L + 1, :L + 1], cmap="viridis")
            axes[h + k].set_title(f"dec cross h{k}")
        fig.suptitle("Attention weights (last layer)")
        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, "attention.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"  已保存注意力热力图: {path}")


def show_predictions(model, n=5):
    src, _, tgt_out, _ = random_batch(n, TASK)
    pred = greedy_decode(model, src)
    tgt_np = tgt_out.cpu().numpy()
    src_np = src.cpu().numpy()
    print("\n  [预测样例]")
    for i in range(n):
        print(f"    输入  : {seq_str(src_np[i])}")
        print(f"    期望  : {seq_str(tgt_np[i])}")
        print(f"    预测  : {seq_str(pred[i])}")


# ============================ 11. 主流程 ============================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 68)
    print(f"任务: {TASK}   |  d_model={D_MODEL}  d_ff={D_FF}  heads={N_HEADS}  layers={N_LAYERS}")
    print(f"词表 {VOCAB_SIZE} 个 token，序列统一长度 {SEQ_LEN}，batch={BATCH_SIZE}，steps={TRAIN_STEPS}")
    print(f"设备: {DEVICE}   |  torch {torch.__version__}")
    print("=" * 68)

    model = Transformer().to(DEVICE)
    print(f"可训练参数: {sum(p.numel() for p in model.parameters()):,}\n")

    lr_schedule = TransformerLRSchedule(D_MODEL, WARMUP_STEPS)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,                       # 真正的 lr 由下面的 LambdaLR 算，这里给 1.0 当乘数
        betas=(0.9, 0.98), eps=1e-9)  # 论文 5.3 的 Adam 超参
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    model.train()
    t0 = time.time()
    for step in range(1, TRAIN_STEPS + 1):
        src, tgt_in, tgt_out, _ = random_batch(BATCH_SIZE)

        logits = model(src, tgt_in)
        loss = masked_loss(logits, tgt_out)

        optimizer.zero_grad()                  # PyTorch 梯度默认累加，每轮必须清
        loss.backward()                        # 反向传播，梯度写进 param.grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 工程实践：防梯度爆炸
        optimizer.step()                       # 参数更新
        scheduler.step()                       # 学习率前进一格

        if step % 100 == 0:
            print(f"step {step:5d}/{TRAIN_STEPS}  loss={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}  耗时 {time.time() - t0:.0f}s")

        if step % 500 == 0:
            print(f"    -> 评估准确率(整句完全正确): {evaluate(model):.4f}")
            model.train()

    print(f"\n训练结束，总耗时 {time.time() - t0:.1f}s")

    print(f"\n最终评估准确率: {evaluate(model, n_batches=10):.4f}")
    show_predictions(model)
    show_attention(model)

    # 长度外推：训练时最长 8 个符号，这里试试 10 个。
    # 论文 3.5 说正弦位置编码理论上能外推，但实际能不能行，取决于训练长度与数据分布。
    print("\n  [长度外推测试] 训练最长 8 个符号，用 10 个符号试试：")
    L = 10
    s = rng.integers(0, N_SYMBOLS, size=L) + 3
    long_src_np = np.append(s, EOS).astype(np.int32)[None, :]
    long_src = torch.as_tensor(long_src_np, dtype=torch.long, device=DEVICE)
    print(f"    输入  : {seq_str(long_src_np[0])}")
    print(f"    期望  : {seq_str(np.append(np.append(SOS, s[::-1]), EOS))}")
    print(f"    预测  : {seq_str(greedy_decode(model, long_src, max_tokens=L + 1)[0])}")

    path = os.path.join(RESULTS_DIR, "transformer.pt")
    torch.save(model.state_dict(), path)
    print(f"\n权重已保存: {path}")


if __name__ == "__main__":
    main()
