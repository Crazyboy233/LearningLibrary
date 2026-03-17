import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

device = "/CPU:0"
batch_size = 64
epochs = 3
data_root = "./data"

experiments = [
    {"name": "mlp_base", "hidden_sizes": [256], "activation": "relu"},
    {"name": "mlp_512", "hidden_sizes": [512], "activation": "relu"},
    {"name": "mlp_1024", "hidden_sizes": [1024], "activation": "relu"},
    {"name": "mlp_two_hidden", "hidden_sizes": [512, 256], "activation": "relu"},
    {"name": "mlp_sigmoid", "hidden_sizes": [256], "activation": "sigmoid"},
    {"name": "cnn_simple", "cnn": True}
]

os.makedirs("results", exist_ok=True)

# ========== 数据 ==========
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., None].astype("float32") / 255.
x_test = x_test[..., None].astype("float32") / 255.

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# ========== 模型 ==========
class MLP(tf.keras.Model):
    def __init__(self, hidden_sizes=[256], activation="relu"):
        super().__init__()
        self.layers_list = []
        self.flatten = tf.keras.layers.Flatten()

        last_dim = 28 * 28
        for h in hidden_sizes:
            self.layers_list.append(tf.keras.layers.Dense(h, activation=activation))
        self.out_layer = tf.keras.layers.Dense(10)

    def call(self, x, return_activations=False, training=False):
        acts = []
        out = self.flatten(x)

        for layer in self.layers_list:
            out = layer(out, training=training)
            acts.append(tf.identity(out))   # 记录激活

        out = self.out_layer(out)
        if return_activations:
            return out, acts
        return out


class SimpleCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10)

    def call(self, x, return_activations=False, training=False):
        acts = []
        x = self.conv1(x)
        acts.append(tf.identity(x))
        x = self.pool1(x)
        acts.append(tf.identity(x))
        x = self.conv2(x)
        acts.append(tf.identity(x))
        x = self.pool2(x)
        acts.append(tf.identity(x))

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        if return_activations:
            return x, acts
        return x

# ========== 评估 ==========
def evaluate(model, ds, loss_fn):
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in ds:
        logits = model(x, training=False)
        loss = loss_fn(y, logits)
        loss_sum += loss.numpy() * x.shape[0]

        preds = tf.argmax(logits, axis=1)
        correct += tf.reduce_sum(tf.cast(preds == y, tf.int32)).numpy()
        total += x.shape[0]

    return loss_sum / total, correct / total

# ========== 训练 ==========
optimizer = tf.keras.optimizers.SGD(0.01)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def train_one_exp(cfg):
    print(f"\n=== Running {cfg['name']} ===")

    if cfg.get("cnn", False):
        model = SimpleCNN()
    else:
        model = MLP(hidden_sizes=cfg["hidden_sizes"], activation=cfg["activation"])

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    logdir = f"./logs/{cfg['name']}"
    os.makedirs(logdir, exist_ok=True)

    tf.profiler.experimental.start(logdir)

    for epoch in range(epochs):
        total = 0
        correct = 0
        loss_sum = 0.0

        for step, (x, y) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = loss_fn(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            preds = tf.argmax(logits, axis=1)
            correct += tf.reduce_sum(tf.cast(preds == y, tf.int32)).numpy()
            loss_sum += loss.numpy() * x.shape[0]
            total += x.shape[0]

            if step >= 10:
                break

        train_loss = loss_sum / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, test_ds, loss_fn)

        print(f"[{cfg['name']}] Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    tf.profiler.experimental.stop()

    print(f"[{cfg['name']}] Done")

if __name__ == "__main__":
    for cfg in experiments:
        train_one_exp(cfg)
