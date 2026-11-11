import pytest
import torch
from safetensors.torch import save_file

from p2 import config, vitrina
from p2.lora import LoraSwitch


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.ModuleDict({"to_k": torch.nn.Linear(6, 4, bias=False)})


def _lora_file(tmp_path, key="unet.attn.to_k"):
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(2, 6, generator=g), torch.randn(4, 2, generator=g)
    f = tmp_path / "l.safetensors"
    save_file({f"{key}.lora_A.weight": a, f"{key}.lora_B.weight": b}, str(f))
    return f, a, b


def test_lora_adds_delta_and_restores_exactly(tmp_path):
    m = Tiny()
    w0 = m.attn.to_k.weight.detach().clone()
    f, a, b = _lora_file(tmp_path)
    sw = LoraSwitch(m, f, scale=1.0)
    sw.set(True)
    assert torch.allclose(m.attn.to_k.weight, w0 + b @ a, atol=1e-6)
    sw.set(True, scale=0.5)                        # reaplicar con otra escala no acumula
    assert torch.allclose(m.attn.to_k.weight, w0 + 0.5 * (b @ a), atol=1e-6)
    sw.set(False)
    assert torch.equal(m.attn.to_k.weight, w0)     # quitado: pesos originales exactos


def test_lora_mismatch_fails_loudly(tmp_path):
    f, _, _ = _lora_file(tmp_path, key="unet.attn.no_existe")
    with pytest.raises(KeyError):
        LoraSwitch(Tiny(), f)


def test_lora_trigger_only_when_enabled():
    cfg = config.load("config/vitrina.yaml")
    assert "vtr_m_camisa_celeste" not in vitrina._look(cfg, ["m_camisa_celeste"])["text"]
    assert vitrina._look(cfg, ["m_camisa_celeste"], lora=True)["text"].startswith("vtr_m_camisa_celeste, ")
