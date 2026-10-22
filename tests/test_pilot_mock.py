"""Orquestación completa del piloto con el backend 'mock' (sin modelos ni GPU)."""
import csv
import json

import pytest

from conftest import ROOT, image_bytes
from experiments import pilot
from p2.export import verify_manifest


@pytest.fixture
def run(tmp_path, cfg):
    inp = tmp_path / "in.png"
    inp.write_bytes(image_bytes(size=(1024, 1024)))
    out = tmp_path / "out"
    res = pilot.generate_all(cfg, inp, out, backend="mock")
    return res, out, inp


def test_generates_exactly_5_per_method_and_records_all(run):
    res, out, _ = run
    recs = [json.loads(l) for l in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    main = [r for r in recs if r["role"] == "main"]
    assert res["n_main"] == 10 and len(main) == 10
    for m in ("A1", "A2"):
        assert sorted(r["variation"] for r in main if r["method"] == m) == ["V1", "V2", "V3", "V4", "V5"]
    assert {r["role"] for r in recs} == {"main", "warmup", "determinism"}
    assert all(r["seed"] in range(1001, 1006) and r["steps"] == 20 and r["strength"] == 0.4 for r in recs)
    assert res["determinism_identical"] is True


def test_resume_does_not_duplicate(run, cfg):
    _, out, inp = run
    pilot.generate_all(cfg, inp, out, backend="mock")
    assert len((out / "runs.jsonl").read_text(encoding="utf-8").splitlines()) == 12


def test_metrics_blind_and_export(run, cfg):
    _, out, _ = run
    rows, sets = pilot.compute_metrics(cfg, out, models=None)
    assert len(rows) == 10 and {s["method"] for s in sets} == {"A1", "A2"}
    with open(out / "metrics.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 10
    key = pilot.contact_sheets(out)
    assert len(key) == 10 and not (out / "blind" / "blind_key.json").exists()
    assert sorted(p.name for p in (out / "blind").glob("B*.png")) == [f"B{i:02d}.png" for i in range(1, 11)]
    n = pilot.finalize(cfg, out, ROOT / "config" / "pilot.yaml")
    assert n > 20 and verify_manifest(out) == []
    assert (out / "LIMITATIONS.md").exists() and json.loads((out / "env.json").read_text())["packages"]


@pytest.mark.parametrize("validation", [None, {"ok": False}, {"ok": "yes"}])
def test_real_backend_blocked_without_valid_validation(tmp_path, cfg, validation, monkeypatch):
    from p2 import env_check, generate
    monkeypatch.setattr(generate, "load_sd15", lambda *a, **k: pytest.fail("no debe cargar SD 1.5"))
    inp = tmp_path / "in.png"
    inp.write_bytes(image_bytes())
    with pytest.raises(env_check.ValidationError):
        pilot.generate_all(cfg, inp, tmp_path / "o", backend="sd15", validation=validation)
    assert not (tmp_path / "o").exists()


def test_real_metrics_blocked_without_validation(run, cfg):
    from p2 import env_check
    _, out, _ = run
    with pytest.raises(env_check.ValidationError):
        pilot.compute_metrics(cfg, out, models=object(), validation=None)


class FakeModels:
    """Sustituto de metrics.Models sin torch: embeddings deterministas y LPIPS ausente."""
    lpips = None

    def img_emb(self, im):
        import numpy as np
        v = np.asarray(im.resize((4, 4)), dtype=float).ravel() + 1
        return v / np.linalg.norm(v)

    def txt_emb(self, text):
        import numpy as np
        v = np.ones(48)
        return v / np.linalg.norm(v)


def test_metrics_without_lpips_are_marked(run, cfg):
    _, out, _ = run
    rows, sets = pilot.compute_metrics(cfg, out, models=FakeModels(), validation={"ok": True})
    assert all(r["lpips"] is None and "clip_img_sim" in r and "clip_delta" in r for r in rows)
    assert all(s["lpips_div"] is None and s["lpips_available"] is False and s["clip_div"] >= 0 for s in sets)


def test_unknown_backend(tmp_path, cfg):
    inp = tmp_path / "in.png"
    inp.write_bytes(image_bytes())
    with pytest.raises(ValueError):
        pilot.generate_all(cfg, inp, tmp_path / "o", backend="otro")


def test_strength_override_warmup_flag_and_sweep_summary(tmp_path, cfg):
    """Piloto 2: un subdirectorio por strength, strength registrado, calentamiento solo en el primero."""
    inp = tmp_path / "in.png"
    inp.write_bytes(image_bytes(size=(1024, 1024)))
    base = tmp_path / "out"
    strengths = cfg["generation"]["strengths"]
    assert strengths == [0.55, 0.70]
    for i, s in enumerate(strengths):
        d = base / f"strength_{s:.2f}"
        pilot.generate_all(cfg, inp, d, backend="mock", strength=s, warmup=(i == 0))
        recs = [json.loads(l) for l in (d / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(r["strength"] == s for r in recs)
        assert sum(r["role"] == "warmup" for r in recs) == (1 if i == 0 else 0)
        pilot.compute_metrics(cfg, d, models=FakeModels(), validation={"ok": True})
    summary = pilot.sweep_summary(base, strengths)
    assert [(r["strength"], r["method"]) for r in summary] == [(0.55, "A1"), (0.55, "A2"), (0.70, "A1"), (0.70, "A2")]
    assert all(r["n"] == 5 and r["clip_delta_mean"] is not None for r in summary)
    assert (base / "sweep_summary.csv").exists()
    assert cfg["generation"]["strength"] == 0.40  # el valor base del piloto 1 no cambia


def test_pipe_is_reused_not_reloaded(tmp_path, cfg, monkeypatch):
    from p2 import generate
    monkeypatch.setattr(generate, "load_sd15", lambda *a, **k: pytest.fail("no debe recargar el pipeline"))
    monkeypatch.setattr(generate, "sd15_generate", lambda pipe, im, spec, gen: (generate.mock_generate(im, spec)[0], False))
    inp = tmp_path / "in.png"
    inp.write_bytes(image_bytes())
    r = pilot.generate_all(cfg, inp, tmp_path / "o", backend="sd15", validation={"ok": True}, strength=0.55, pipe=object())
    assert r["n_main"] == 10
