"""Regression test for a serialization bug in `swiss_army_keras.activations`.

`GELU`/`Snake` were never registered via Keras's serialization registry
(`tf.keras.utils.register_keras_serializable`). That's invisible for a
standalone model, but breaks - `TypeError: 'str' object is not callable' -
specifically when the layer is used inside a sub-model that itself gets
called as a layer within an outer model (`inner_model([outer_input])`,
exactly the pattern `backbone_zoo()` uses for every backbone) and that outer
model is then cloned, e.g. by `tfmot.quantization.keras.quantize_apply`
during QAT. No existing backbone used `GELU` before RepViT (plan.md §2.4),
so this had never been exercised. Fixed by adding the registration
decorator; this test pins the fix.
"""

import tensorflow as tf

from swiss_army_keras.activations import GELU, Snake


def _nested_submodel_clones(layer):
    inp = tf.keras.Input((8, 8, 4))
    x = layer(inp)
    inner = tf.keras.Model(inp, x, name="inner")

    outer_inp = tf.keras.Input((8, 8, 4))
    y = inner([outer_inp])
    outer = tf.keras.Model(outer_inp, y)

    tf.keras.models.clone_model(outer)  # raises if unregistered


def test_gelu_survives_nested_submodel_clone():
    _nested_submodel_clones(GELU(name="g"))


def test_snake_survives_nested_submodel_clone():
    _nested_submodel_clones(Snake(name="s"))
