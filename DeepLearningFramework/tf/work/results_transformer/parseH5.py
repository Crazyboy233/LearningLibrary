import h5py, numpy as np

P = "results_transformer/transformer.weights.h5"

with h5py.File(P, "r") as f:
    # 结构：递归打印所有叶子节点的名字/形状/dtype
    f.visititems(lambda n, o: print(f"{n:<55}{str(o.shape):<14}{o.dtype}")
                 if isinstance(o, h5py.Dataset) else None)

    # 数值：注意要写 [:] 才是真的读进来，不加只能拿到 Dataset 对象
    W = f["encoder_embedding/vars/0"][:]
    print(W.shape, W.dtype, W[:2, :4])
