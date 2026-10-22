import io
import json

from tools.select_images import KAGGLE_LIST, candidates, candidates_from_names, fetch_sample_names, main


def row(fn, model, preset="photographic"):
    return {"image_filename": fn, "model_used": model, "text_prompt": "a face",
            "configs": json.dumps({"style_preset": preset, "engine_id": "stable-diffusion-xl-1024-v1-0"})}


ROWS = [
    row("SDXL_image_0000001.jpg", "SDXL"),
    row("SDXL_image_0000002.jpg", "SDXL", "enhance"),        # no fotográfico
    row("SDXL_image_0000003.jpg", "FLUX1_dev"),             # prefijo y model_used no coinciden
    row("FLUX1_dev_image_0000001.jpg", "FLUX1_dev"),
    row("DALLE3_image_0000001.jpg", "DALLE3"),
    row("SDXL_image_0000004.jpg", "SDXL", "cinematic"),
    row("raro.jpg", "SDXL"),
]
NAMES = ["SDXL_image_0000004.jpg", "SDXL_image_0000001.jpg", "FLUX1_dev_image_0000001.jpg",
         "DALLE3_image_0000001.jpg", "raro.jpg", "SDXL_image_0000001.jpg"]


def test_only_consistent_sdxl_photographic():
    names = [c["image_filename"] for c in candidates(ROWS)]
    assert sorted(names) == ["SDXL_image_0000001.jpg", "SDXL_image_0000004.jpg"]


def test_sample_list_filter_and_determinism():
    a = candidates(ROWS, sample_names={"SDXL_image_0000004.jpg"})
    assert [c["image_filename"] for c in a] == ["SDXL_image_0000004.jpg"]
    assert candidates(ROWS) == candidates(ROWS)


def test_flux_only_if_allowed():
    names = [c["image_filename"] for c in candidates(ROWS, models=("FLUX1_dev",))]
    assert names == ["FLUX1_dev_image_0000001.jpg"]


def test_names_mode_without_csv_is_deterministic_and_filters():
    c = candidates_from_names(NAMES)
    assert sorted(x["image_filename"] for x in c) == ["SDXL_image_0000001.jpg", "SDXL_image_0000004.jpg"]
    assert c == candidates_from_names(list(reversed(NAMES)))  # no depende del orden de entrada
    assert all(x["style_preset"] is None for x in c)


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_sample_names_paginates_without_network():
    pages = {KAGGLE_LIST: {"datasetFiles": [{"name": "a_small_sample_new/SDXL_image_0000001.jpg"},
                                            {"name": "SFHQ_T2I_dataset.csv"}], "nextPageTokenNullable": "t/2+"},
             KAGGLE_LIST + "&pageToken=t%2F2%2B": {"datasetFiles": [{"name": "a_small_sample_new/SDXL_image_0000004.jpg"}],
                                                   "nextPageTokenNullable": None}}
    calls = []

    def opener(url, timeout):
        calls.append(url)
        return FakeResp(json.dumps(pages[url]).encode())
    assert fetch_sample_names(opener) == ["SDXL_image_0000001.jpg", "SDXL_image_0000004.jpg"]
    assert len(calls) == 2


def test_cli_survives_cp1252_console(tmp_path):
    """Regresión: en Windows con salida redirigida (cp1252) imprimir «→» lanzaba UnicodeEncodeError."""
    import os
    import subprocess
    import sys
    from conftest import ROOT
    f = tmp_path / "names.txt"
    f.write_text("\n".join(NAMES), encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "select_images.py"), "--sample-list", str(f)],
                       capture_output=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr.decode("cp1252", "replace")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "register_image.py"), "--help"],
                       capture_output=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr.decode("cp1252", "replace")


def test_cli_with_sample_list_file(tmp_path, capsys):
    f = tmp_path / "names.txt"
    f.write_text("\n".join("a_small_sample_new/" + n for n in NAMES), encoding="utf-8")
    cands = main(["--sample-list", str(f), "--n", "5"])
    out = capsys.readouterr().out
    assert len(cands) == 2 and "sin CSV" in out and "Descargue SOLO" in out
