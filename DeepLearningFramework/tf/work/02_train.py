"""
Transformer（Attention Is All You Need）从零实现 + 训练示例

任务：合成序列任务（默认“序列反转”，可选“原样复制”）
      输入  [3 7 1 5] EOS   ->  输出  SOS [5 1 7 3] EOS

为什么用这个任务：不需要下载任何数据集，CPU 上几分钟就能训到接近满分，
而且注意力矩阵会呈现漂亮的“反对角线 / 对角线”，能直观看到模型真的学会了位置对齐。

本文件与论文的对应关系（括号内为论文小节号）：
    1. 配置                      —— 表 3 base 列（这里按 CPU 规模缩小）
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

运行（先激活虚拟环境）：
    cd DeepLearningFramework/tf
    source tf-env/bin/activate
    python work/02_train.py
"""
import os
import time
import numpy as np
import tensorflow as tf

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
SEQ_LEN = MAX_SYMBOLS + 1        # 统一补齐到 9；形状固定 => tf.function 不必反复重追踪

# 论文 base 模型用 D_MODEL=512 / D_FF=2048 / N_HEADS=8 / N_LAYERS=6，
# 这里为 CPU 演示缩小，重点是把结构讲清楚。
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

    return src, tgt_in, tgt_out, lens


def sym_name(idx):
    """把 token id 打印成人能看懂的样子。"""
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
        if i == PAD or i == EOS:
            break
        out.append(sym_name(i))
    return out


# ============================ 3. 掩码 ============================
NEG_INF = -1e9


def padding_mask(seq):
    """padding 掩码：形状 (B, 1, 1, T)，PAD 位置为 -1e9，其余为 0（加性掩码）。"""
    keep = tf.cast(tf.not_equal(seq, PAD), tf.float32)
    return (1.0 - keep[:, tf.newaxis, tf.newaxis, :]) * NEG_INF


def causal_mask(size):
    """look-ahead 掩码（论文 3.2.3）：下三角为 0、上三角为 -1e9。

    位置 i 只能看到 <= i 的位置，从而保证解码器的自回归性质。
    """
    tri = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return (1.0 - tri)[tf.newaxis, tf.newaxis, :, :] * NEG_INF


# ============================ 4. 缩放点积注意力 + 多头 ============================
class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHead(Q,K,V) = Concat(head_1..head_h) W^O
       head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
       Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
    """

    def __init__(self, d_model, n_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads          # 论文 3.2.2：d_k = d_v = d_model / h
        self.d_v = d_model // n_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wo = tf.keras.layers.Dense(d_model)   # W^O
        self.attn_dropout = tf.keras.layers.Dropout(dropout)

    def _split_heads(self, x):
        """(B, T, d_model) -> (B, n_heads, T, d_k)"""
        B = tf.shape(x)[0]
        x = tf.reshape(x, (B, -1, self.n_heads, self.d_k))
        return tf.transpose(x, (0, 2, 1, 3))

    def call(self, q, k, v, attn_mask=None, training=False):
        # 注意：参数名不能叫 `mask`——那是 Keras Layer 内置的掩码机制保留字，
        # 同名会被框架接管并产生 "does not support masking" 警告。
        B = tf.shape(q)[0]

        q = self._split_heads(self.wq(q))       # (B, h, Tq, d_k)
        k = self._split_heads(self.wk(k))       # (B, h, Tk, d_k)
        v = self._split_heads(self.wv(v))       # (B, h, Tk, d_v)

        # 式(1)：先点积、再除以 sqrt(d_k) —— d_k 越大点积数值越大，
        # 不缩放会让 softmax 落入梯度极小的饱和区（论文 3.2.1 脚注 4）。
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.d_k, tf.float32))      # (B, h, Tq, Tk)

        if attn_mask is not None:
            scores = scores + attn_mask         # 掩掉的位置 softmax 后权重 ~ 0

        attn = tf.nn.softmax(scores, axis=-1)   # 每行归一化成权重分布
        attn = self.attn_dropout(attn, training=training)

        out = tf.matmul(attn, v)                # (B, h, Tq, d_v) 值的加权和
        out = tf.transpose(out, (0, 2, 1, 3))   # (B, Tq, h, d_v)
        out = tf.reshape(out, (B, -1, self.d_model))
        return self.wo(out), attn               # 拼接后过一次线性投影


# ============================ 5. 位置编码 ============================
class PositionalEncoding(tf.keras.layers.Layer):
    """论文 3.5：PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(...)

    用固定公式而不是可学习参数，好处是可以外推到训练时没见过的更长的序列（表 3 (E)）。
    """

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def call(self, x):
        T = tf.shape(x)[1]
        pos = tf.cast(tf.range(T), tf.float32)[:, tf.newaxis]        # (T, 1)
        i = tf.cast(tf.range(self.d_model), tf.float32)[tf.newaxis, :]  # (1, d)
        angle = pos / tf.pow(10000.0, (2 * (i // 2)) / float(self.d_model))
        even = tf.equal(tf.cast(i, tf.int32) % 2, 0)                 # 偶数列 sin、奇数列 cos
        pe = tf.where(even, tf.sin(angle), tf.cos(angle))            # (T, d)
        return x + pe[tf.newaxis, :, :]


# ============================ 6. 前馈网络 / 编码器层 / 解码器层 ============================
def pointwise_ffn(d_model, d_ff, dropout):
    """式(2)：FFN(x) = ReLU(x W1 + b1) W2 + b2，对每个位置独立且共享参数。"""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation="relu"),
        tf.keras.layers.Dense(d_model),
        tf.keras.layers.Dropout(dropout),
    ])


class EncoderLayer(tf.keras.layers.Layer):
    """论文 3.1 编码器层：自注意力 + FFN，两个子层都是 LayerNorm(x + Sublayer(x))。"""

    def __init__(self, d_model, d_ff, n_heads, dropout, **kwargs):
        super().__init__(**kwargs)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = pointwise_ffn(d_model, d_ff, dropout)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, src_mask, training=False):
        h, attn = self.self_attn(x, x, x, attn_mask=src_mask, training=training)
        x = self.ln1(x + self.drop1(h, training=training))     # 残差 + LayerNorm
        h = self.ffn(x, training=training)
        x = self.ln2(x + self.drop2(h, training=training))
        return x, attn


class DecoderLayer(tf.keras.layers.Layer):
    """论文 3.1 解码器层：比编码器多一个「编码器-解码器注意力」子层（共 3 个子层）。"""

    def __init__(self, d_model, d_ff, n_heads, dropout, **kwargs):
        super().__init__(**kwargs)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)   # 带 look-ahead 掩码
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)  # Q 来自解码器，K/V 来自编码器
        self.ffn = pointwise_ffn(d_model, d_ff, dropout)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)
        self.drop3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_out, src_mask, tgt_mask, training=False):
        h, self_attn = self.self_attn(x, x, x, attn_mask=tgt_mask, training=training)
        x = self.ln1(x + self.drop1(h, training=training))
        h, cross_attn = self.cross_attn(x, enc_out, enc_out, attn_mask=src_mask, training=training)
        x = self.ln2(x + self.drop2(h, training=training))
        h = self.ffn(x, training=training)
        x = self.ln3(x + self.drop3(h, training=training))
        return x, self_attn, cross_attn


# ============================ 7. Transformer ============================
class Transformer(tf.keras.Model):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, d_ff=D_FF,
                 n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model

        self.encoder_embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.decoder_embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.emb_dropout = tf.keras.layers.Dropout(dropout)

        self.enc_layers = [EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)]
        self.dec_layers = [DecoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)]

    def encode(self, src, src_mask, training=False):
        x = self.encoder_embedding(src) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # 3.4：乘 sqrt(d_model)
        x = self.emb_dropout(self.pos_encoding(x), training=training)
        attns = []
        for layer in self.enc_layers:
            x, a = layer(x, src_mask, training=training)
            attns.append(a)
        return x, attns

    def decode(self, tgt_in, enc_out, src_mask, tgt_mask, training=False):
        x = self.decoder_embedding(tgt_in) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.emb_dropout(self.pos_encoding(x), training=training)
        self_attns, cross_attns = [], []
        for layer in self.dec_layers:
            x, sa, ca = layer(x, enc_out, src_mask, tgt_mask, training=training)
            self_attns.append(sa)
            cross_attns.append(ca)
        return x, self_attns, cross_attns

    def output_projection(self, x):
        """3.4：输出投影与嵌入层共享权重的转置矩阵（此处直接复用同一个变量，
        GradientTape 会自动把两处的梯度累加到同一份参数上），无需 bias。
        """
        return tf.matmul(x, self.encoder_embedding.embeddings, transpose_b=True)

    def call(self, src, tgt_in, training=False, return_attention=False):
        src_mask = padding_mask(src)                                                   # 屏蔽输入 PAD
        tgt_mask = padding_mask(tgt_in) + causal_mask(tf.shape(tgt_in)[1])              # PAD + 未来位置

        enc_out, enc_attns = self.encode(src, src_mask, training=training)
        dec_out, dec_self_attns, dec_cross_attns = self.decode(
            tgt_in, enc_out, src_mask, tgt_mask, training=training)
        logits = self.output_projection(dec_out)

        if return_attention:
            return logits, {"enc_self": enc_attns, "dec_self": dec_self_attns, "dec_cross": dec_cross_attns}
        return logits


# ============================ 8. 损失与学习率 ============================
def masked_loss(logits, labels):
    """标签平滑（5.4）+ 只在真实 token 上求平均（忽略 PAD 位置）。

    label_smoothing=0.1 会让模型学到“不那么自信”的分布，
    困惑度（PPL）变差但翻译准确率/BLEU 反而提升。
    """
    y_true = tf.one_hot(labels, VOCAB_SIZE)
    per_token = tf.keras.losses.categorical_crossentropy(
        y_true, logits, from_logits=True, label_smoothing=0.1)      # (B, T)
    mask = tf.cast(tf.not_equal(labels, PAD), tf.float32)           # (B, T)
    return tf.reduce_sum(per_token * mask) / tf.reduce_sum(mask)


class TransformerLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """论文 5.3 式(3)：lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)

    前 warmup_steps 步线性升温，之后按步数平方根倒数衰减。
    这种“先热身再退火”对 Post-LN 的 Transformer 尤其关键。
    """

    def __init__(self, d_model, warmup_steps):
        super().__init__()
        self.d_model = float(d_model)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = tf.maximum(tf.cast(step, tf.float32), 1.0)
        return self.d_model ** -0.5 * tf.minimum(step ** -0.5, step * self.warmup_steps ** -1.5)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}


@tf.function
def train_step(model, optimizer, src, tgt_in, tgt_out):
    with tf.GradientTape() as tape:
        logits = model(src, tgt_in, training=True)
        loss = masked_loss(logits, tgt_out)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)      # 工程实践：防止偶发梯度爆炸
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# ============================ 9. 自回归贪心解码 ============================
def greedy_decode(model, src, max_tokens=SEQ_LEN):
    """逐 token 生成：每步只把已生成的部分喂给解码器，取概率最大的下一个 token。"""
    src = tf.constant(src, tf.int32)
    src_mask = padding_mask(src)
    enc_out, _ = model.encode(src, src_mask, training=False)

    B = tf.shape(src)[0]
    ys = tf.constant(np.full((B, 1), SOS, np.int32), tf.int32)
    for _ in range(max_tokens):
        tgt_mask = causal_mask(tf.shape(ys)[1])                       # 生成时没有 PAD
        dec_out, _, _ = model.decode(ys, enc_out, src_mask, tgt_mask, training=False)
        logits = model.output_projection(dec_out)[:, -1, :]           # 只关心最后一个位置
        nxt = tf.argmax(logits, axis=-1, output_type=tf.int32)
        ys = tf.concat([ys, nxt[:, tf.newaxis]], axis=1)
    return ys[:, 1:].numpy()                                          # 去掉开头那个 SOS


def evaluate(model, n_batches=6):
    """指标：整条序列完全正确才算对（exact match）。"""
    correct = total = 0
    for _ in range(n_batches):
        src, _, tgt_out, _ = random_batch(BATCH_SIZE, TASK)
        pred = greedy_decode(model, src)
        for i in range(len(src)):
            correct += int(seq_tokens(tgt_out[i]) == seq_tokens(pred[i]))
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
    if src is None:                      # 固定一条样本，方便观察对齐模式
        s = np.array([2, 7, 0, 5, 9, 3], np.int32) + 3
        src = np.full((1, SEQ_LEN), PAD, np.int32)
        src[0, :len(s) + 1] = np.append(s, EOS)

    L = int(np.sum(src[0] != PAD)) - 1                       # 真实符号数（去掉 EOS）
    rev = (src[0][:L] - 3)                                   # 目标符号 = 源符号倒序
    tgt_in = np.full((1, SEQ_LEN), PAD, np.int32)
    tgt_in[0, :L + 1] = np.append(SOS, rev[::-1] + 3)        # 解码器输入：SOS + 倒序目标

    _, info = model(tf.constant(src, tf.int32), tf.constant(tgt_in, tf.int32),
                    training=False, return_attention=True)
    enc_self = info["enc_self"][-1][0].numpy()               # (h, T, T)
    dec_cross = info["dec_cross"][-1][0].numpy()             # (h, Tq, Tk)

    src_lab = [_sym_short(v) for v in src[0][:L + 1]]
    tgt_lab = [_sym_short(v) for v in tgt_in[0][:L + 1]]

    print("\n  ===== 注意力可视化（第 0 个样本）=====")
    print(f"  源序列: {seq_str(src[0])}")
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
    print("\n  [预测样例]")
    for i in range(n):
        print(f"    输入  : {seq_str(src[i])}")
        print(f"    期望  : {seq_str(tgt_out[i])}")
        print(f"    预测  : {seq_str(pred[i])}")


# ============================ 11. 主流程 ============================
def main():
    tf.keras.utils.set_random_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 68)
    print(f"任务: {TASK}   |  d_model={D_MODEL}  d_ff={D_FF}  heads={N_HEADS}  layers={N_LAYERS}")
    print(f"词表 {VOCAB_SIZE} 个 token，序列统一长度 {SEQ_LEN}，batch={BATCH_SIZE}，steps={TRAIN_STEPS}")
    print("=" * 68)

    model = Transformer()
    # 先跑一次前向，把参数建出来才能统计参数量
    dummy_src, dummy_tgt, _, _ = random_batch(2)
    _ = model(dummy_src, dummy_tgt)
    print(f"可训练参数: {sum(int(np.prod(v.shape)) for v in model.trainable_variables):,}\n")

    lr_schedule = TransformerLRSchedule(D_MODEL, WARMUP_STEPS)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9, beta_2=0.98, epsilon=1e-9)      # 论文 5.3 的 Adam 超参

    t0 = time.time()
    for step in range(1, TRAIN_STEPS + 1):
        src, tgt_in, tgt_out, _ = random_batch(BATCH_SIZE)
        loss = train_step(model, optimizer, src, tgt_in, tgt_out)

        if step % 100 == 0:
            lr = float(lr_schedule(tf.cast(step, tf.float32)))
            print(f"step {step:5d}/{TRAIN_STEPS}  loss={float(loss):.4f}  lr={lr:.6f}  "
                  f"耗时 {time.time() - t0:.0f}s")

        if step % 500 == 0:
            print(f"    -> 评估准确率(整句完全正确): {evaluate(model):.4f}")

    print(f"\n训练结束，总耗时 {time.time() - t0:.1f}s")

    print(f"\n最终评估准确率: {evaluate(model, n_batches=10):.4f}")
    show_predictions(model)
    show_attention(model)

    # 长度外推：训练时最长 8 个符号，这里试试 10 个。
    # 论文 3.5 说正弦位置编码理论上能外推，但实际能不能行，取决于训练长度与数据分布。
    print("\n  [长度外推测试] 训练最长 8 个符号，用 10 个符号试试：")
    L = 10
    long_src = np.full((1, L + 1), PAD, np.int32)
    s = rng.integers(0, N_SYMBOLS, size=L) + 3
    long_src[0, :L + 1] = np.append(s, EOS)
    print(f"    输入  : {seq_str(long_src[0])}")
    print(f"    期望  : {seq_str(np.append(np.append(SOS, s[::-1]), EOS))}")
    print(f"    预测  : {seq_str(greedy_decode(model, long_src, max_tokens=L + 1)[0])}")

    path = os.path.join(RESULTS_DIR, "transformer.weights.h5")
    model.save_weights(path)
    print(f"\n权重已保存: {path}")


if __name__ == "__main__":
    main()
