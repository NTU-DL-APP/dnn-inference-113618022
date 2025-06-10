import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = np.array(x)
    if x.ndim == 1:
        x = x - np.max(x)
        exps = np.exp(x)
        return exps / np.sum(exps)
    elif x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims=True)
    else:
        raise ValueError("Input to softmax must be 1D or 2D numpy array.")

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# === Forward pass using architecture + weights ===
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)
    return x

# === Unified inference entry point ===
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
