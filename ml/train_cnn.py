"""Trains classifier 1: the lightweight 1D CNN, quantises it to int8 and exports it.

    python -m ml.train_cnn
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

from .common import MODELS, RESULTS, load_split, metrics, sample_weights
from .core import CLASSES, N_CTX, ROOT, WAVE_LEN

MAX_EPOCHS = 40
PATIENCE = 8
BATCH = 256
HEADER = ROOT / "firmware" / "src" / "model_cnn.h"
TFLITE = MODELS / "cnn_int8.tflite"
tf.keras.utils.set_random_seed(7)


def build_model(batch_size=None):
    L = tf.keras.layers
    wave_in = L.Input((WAVE_LEN, 1), batch_size=batch_size, name="wave")
    ctx_in = L.Input((N_CTX,), batch_size=batch_size, name="ctx")
    x = wave_in
    for filters, kernel, stride in [(8, 9, 2), (16, 5, 1), (24, 3, 1)]:
        x = L.Conv1D(filters, kernel, strides=stride, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU(6.0)(x)
        x = L.MaxPooling1D(2)(x)
    x = L.Reshape((10 * 24,))(x)  # Fixed shape: Flatten would add dynamic-shape ops on the MCU
    x = L.Dropout(0.25)(x)
    x = L.Dense(24)(x)
    x = L.ReLU(6.0)(x)
    c = L.Dense(16)(ctx_in)
    c = L.ReLU(6.0)(c)
    x = L.Concatenate()([x, c])
    x = L.Dense(32)(x)
    x = L.ReLU(6.0)(x)
    x = L.Dropout(0.25)(x)
    out = L.Dense(len(CLASSES), activation="softmax", name="class")(x)
    return tf.keras.Model([wave_in, ctx_in], out)


class Batches(tf.keras.utils.PyDataset):
    """Weighted batches with light augmentation matching likely device differences."""

    def __init__(self, d, steps, seed=0):
        super().__init__(workers=1, use_multiprocessing=False)
        self.w, self.c, self.y = d["wave"], d["ctx"], d["y"]
        w = sample_weights(self.y, d["rec"])
        self.prob = w / w.sum()
        self.steps = steps
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.steps

    def __getitem__(self, _):
        rng = self.rng
        pick = rng.choice(len(self.y), BATCH, p=self.prob)
        w = self.w[pick].copy()
        shift = rng.integers(-3, 4, len(pick))
        for k in np.flatnonzero(shift):
            w[k] = np.roll(w[k], shift[k])
        w += rng.normal(0, 1, (len(pick), 1)) * 0.02 * rng.normal(0, 1, w.shape)      # ADC and muscle noise
        t = np.linspace(0, 1, WAVE_LEN)[None, :]
        w += rng.normal(0, 0.03, (len(pick), 1)) * np.sin(2 * np.pi * (t + rng.random((len(pick), 1))))  # Baseline
        w *= rng.uniform(0.85, 1.15, (len(pick), 1))
        c = self.c[pick].copy()
        return (w[..., None].astype(np.float32), c), self.y[pick]


class MacroF1(tf.keras.callbacks.Callback):
    def __init__(self, val):
        super().__init__()
        self.x = (val["wave"][..., None], val["ctx"])
        self.y = val["y"]
        self.best, self.best_epoch, self.best_w = -1, 0, None

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.x, batch_size=2048, verbose=0).argmax(1)
        f1 = metrics(self.y, p)["macro_f1"]
        print(f"epoch {epoch + 1}: loss {logs['loss']:.4f} val macro-F1 {f1:.4f}", flush=True)
        if f1 > self.best:
            self.best, self.best_epoch, self.best_w = f1, epoch + 1, self.model.get_weights()
        elif epoch + 1 - self.best_epoch >= PATIENCE:
            self.model.stop_training = True


def fit(train, val, epochs):
    m = build_model()
    steps = 400
    sched = tf.keras.optimizers.schedules.CosineDecay(1e-3, steps * epochs, alpha=0.02)
    m.compile(tf.keras.optimizers.Adam(sched), "sparse_categorical_crossentropy")
    cbs = []
    if val is not None:
        cbs = [MacroF1(val)]
    m.fit(Batches(train, steps), epochs=epochs, callbacks=cbs, verbose=0)
    return m, (cbs[0] if cbs else None)


def quantise(model, data):
    """Full int8 conversion, calibrated on a class-balanced sample of training beats."""
    rng = np.random.default_rng(1)
    pick = np.concatenate([rng.choice(np.flatnonzero(data["y"] == c), min(150, int((data["y"] == c).sum())), replace=False)
                           for c in range(len(CLASSES))])

    def rep():
        for i in pick:
            yield [data["wave"][i][None, :, None].astype(np.float32), data["ctx"][i][None].astype(np.float32)]

    # A fixed batch size keeps dynamic-shape ops out of the graph for TFLite Micro
    fixed = build_model(batch_size=1)
    fixed.set_weights(model.get_weights())
    conv = tf.lite.TFLiteConverter.from_keras_model(fixed)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    # TFLite Micro's fully connected kernel supports one scale per tensor. Without this flag the
    # converter writes per-channel Dense weights, which the MCU silently misreads.
    conv._experimental_disable_per_channel_quantization_for_dense_layers = True
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


class Int8Model:
    def __init__(self, path):
        # The reference kernels are the ones TFLite Micro uses, so scores match the device
        self.i = tf.lite.Interpreter(model_path=str(path),
                                     experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        self.i.allocate_tensors()
        self.inp = {("wave" if d["shape"][1] == WAVE_LEN else "ctx"): d for d in self.i.get_input_details()}
        self.out = self.i.get_output_details()[0]

    def _q(self, d, v):
        s, z = d["quantization"]
        return np.clip(np.round(v / s + z), -128, 127).astype(np.int8)

    def predict(self, wave, ctx):
        res = np.zeros(len(wave), np.int8)
        scores = np.zeros((len(wave), len(CLASSES)), np.int8)
        for k in range(len(wave)):
            self.i.set_tensor(self.inp["wave"]["index"], self._q(self.inp["wave"], wave[k][None, :, None]))
            self.i.set_tensor(self.inp["ctx"]["index"], self._q(self.inp["ctx"], ctx[k][None]))
            self.i.invoke()
            scores[k] = self.i.get_tensor(self.out["index"])[0]
            res[k] = scores[k].argmax()
        return res, scores


def write_header(blob, path):
    rows = [",".join(str(b) for b in blob[i:i + 20]) for i in range(0, len(blob), 20)]
    path.write_text("// Generated by ml/train_cnn.py. Do not edit.\n#pragma once\n#include <stdint.h>\n\n"
                    f"alignas(16) static const uint8_t cnn_model[{len(blob)}] = {{\n" + ",\n".join(rows) + "\n};\n"
                    f"constexpr unsigned int cnn_model_len = {len(blob)};\n")


def main():
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    tr, va = load_split("train"), load_split("val")
    _, cb = fit(tr, va, MAX_EPOCHS)
    print(f"best epoch {cb.best_epoch} val macro-F1 {cb.best:.4f}", flush=True)

    all_tr = {k: np.concatenate([tr[k], va[k]]) for k in tr}
    model, _ = fit(all_tr, None, cb.best_epoch)
    model.save(MODELS / "cnn.keras")
    blob = quantise(model, all_tr)
    TFLITE.write_bytes(blob)
    write_header(blob, HEADER)

    info = {"best_epoch": cb.best_epoch, "val_macro_f1": cb.best, "params": int(model.count_params()),
            "tflite_kb": len(blob) / 1024}
    (RESULTS / "cnn_training.json").write_text(json.dumps(info, indent=1))
    print(info)
    model.summary()


if __name__ == "__main__":
    main()
