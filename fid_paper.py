"""Paper-standard FID / Inception Score network, loaded inside TF2.

Ho et al. (and the FID/IS papers before them) score images with the original 2015
Inception graph (`inception-2015-12-05`), not Keras' ImageNet InceptionV3. This
loads that frozen graph directly, so no PyTorch or TF-GAN is needed.

`PaperInceptionScorer` has the same call signature as `InceptionScorer` in
ddpm_paper.py: scorer(uint8_images) -> (pool features [N, 2048], class probs [N, 1008]),
so the FID / IS math there is reused unchanged.

Notes on matching the published protocol:
  * Images are fed as raw float 0..255 into the graph's own `ExpandDims:0`, so the
    graph performs its own bilinear resize to 299x299 and its own normalization.
  * IS class probabilities come from softmax(pool_3 @ W) using the graph's final
    weight matrix, without the bias term, as in the original Inception Score code.
  * The graph hard-codes batch size 1 in a few static shapes; feeding through a
    `[None, None, None, 3]` placeholder (via `input_map`) avoids that.
"""
import os
import tarfile
import urllib.request

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, graph_pb2

URL ="http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
MEMBER = "classify_image_graph_def.pb"
CACHE = os.path.expanduser("~/.cache/fid_paper")


def ensure_graph(cache=CACHE):
    """Download the archive once and keep only the ~95MB frozen graph."""
    pb = os.path.join(cache, MEMBER)
    if os.path.exists(pb):
        return pb
    os.makedirs(cache, exist_ok=True)
    tgz = os.path.join(cache, "inception-2015-12-05.tgz")
    print(f"downloading {URL}", flush=True)
    urllib.request.urlretrieve(URL, tgz)
    with tarfile.open(tgz) as tar:
        src = tar.extractfile(MEMBER)
        tmp = pb + ".tmp"
        with open(tmp, "wb") as dst:
            dst.write(src.read())
    os.replace(tmp, pb)
    os.remove(tgz)
    return pb


def _float_type():
    return attr_value_pb2.AttrValue(type=tf.float32.as_datatype_enum)


def replace_batchnorm(graph_def):
    """Rewrite the deleted BatchNormWithGlobalNormalization op into basic ops.

    TF2 refuses to create that op (removed in GraphDef version 9), but it is
    just  y = (t - mean) * rsqrt(var + eps) * gamma + beta  (gamma only when
    scale_after_normalization is set), so each node is replaced by an equivalent
    sub-graph whose final node keeps the original name and output.
    """
    out = graph_pb2.GraphDef()
    out.versions.CopyFrom(graph_def.versions)
    out.library.CopyFrom(graph_def.library)

    def add(name, op, inputs):
        n = out.node.add()
        n.name, n.op = name, op
        n.input.extend(inputs)
        n.attr["T"].CopyFrom(_float_type())

    for node in graph_def.node:
        if node.op != "BatchNormWithGlobalNormalization":
            out.node.add().CopyFrom(node)
            continue
        t, mean, var, beta, gamma = list(node.input)[:5]
        name = node.name
        eps = out.node.add()
        eps.name, eps.op = name + "/eps", "Const"
        eps.attr["dtype"].CopyFrom(_float_type())
        eps.attr["value"].tensor.CopyFrom(
            tf.make_tensor_proto(node.attr["variance_epsilon"].f, dtype=tf.float32))
        add(name + "/var_eps", "Add", [var, name + "/eps"])
        add(name + "/rsqrt", "Rsqrt", [name + "/var_eps"])
        mult = name + "/rsqrt"
        if node.attr["scale_after_normalization"].b:
            add(name + "/mult", "Mul", [mult, gamma])
            mult = name + "/mult"
        add(name + "/x_mult", "Mul", [t, mult])
        add(name + "/m_mult", "Mul", [mean, mult])
        add(name + "/sub", "Sub", [name + "/x_mult", name + "/m_mult"])
        add(name, "Add", [name + "/sub", beta])
    return out


class PaperInceptionScorer:
    def __init__(self, pb_path=None):
        pb = pb_path or ensure_graph()
        graph_def = tf.compat.v1.GraphDef()
        with open(pb, "rb") as f:
            graph_def.ParseFromString(f.read())
        graph_def = replace_batchnorm(graph_def)

        def build():
            x = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name="fid_input")
            tf.import_graph_def(graph_def, input_map={"ExpandDims:0": x}, name="")

        wrapped = tf.compat.v1.wrap_function(build, [])
        g = wrapped.graph
        self._features = wrapped.prune(
            feeds=g.get_tensor_by_name("fid_input:0"),
            fetches=g.get_tensor_by_name("pool_3:0"))
        w = g.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        self._w = wrapped.prune(feeds=[], fetches=w)().numpy().astype(np.float64)

    def __call__(self, imgs_uint8, batch=250):
        feats = []
        for i in range(0, len(imgs_uint8), batch):
            x = tf.constant(np.asarray(imgs_uint8[i:i + batch], dtype=np.float32))
            feats.append(self._features(x).numpy().reshape(len(x), -1))
        feats = np.concatenate(feats).astype(np.float64)
        logits = feats @ self._w
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return feats, probs
