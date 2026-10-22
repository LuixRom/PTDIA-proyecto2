"""Compatibilidad de las métricas con transformers 4.x y 5.x (sin torch ni modelos)."""
import types

from p2.metrics import clip_features


def test_clip_features_accepts_tensor_and_v5_output():
    tensor = object()  # transformers 4.x: get_image_features devuelve directamente el tensor
    assert clip_features(tensor) is tensor
    v5 = types.SimpleNamespace(pooler_output="proyeccion", last_hidden_state="otra cosa")  # transformers 5.x
    assert clip_features(v5) == "proyeccion"
