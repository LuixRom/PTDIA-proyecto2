import csv
import json
import re
import shutil
from datetime import date

import pytest
import yaml
from PIL import Image

from conftest import ROOT
from p2.dataset import MANIFEST_FIELDS, check_dataset_dir, model_from_name
from p2.export import sha256_file
from tools import make_zip
from tools.register_image import LICENSE_NOTE, RegistrationError, register

SDXL = "SDXL_image_0001234.jpg"


@pytest.fixture
def proj(tmp_path):
    """Proyecto temporal aislado: config real con pilot_image vacío y manifiesto solo con cabecera
    (independiente de lo que esté registrado en el proyecto real)."""
    (tmp_path / "config").mkdir()
    text = (ROOT / "config" / "pilot.yaml").read_text(encoding="utf-8")
    text = re.sub(r"^(\s+pilot_image:\s*)\S+", r"\1null", text, flags=re.M)
    (tmp_path / "config" / "pilot.yaml").write_text(text, encoding="utf-8")
    d = tmp_path / "data" / "sfhq_t2i"
    d.mkdir(parents=True)
    (d / "MANIFEST.csv").write_text(",".join(MANIFEST_FIELDS) + "\n", encoding="utf-8")
    for f in ("SOURCE.md", "LICENSE_MIT.txt"):
        shutil.copy(ROOT / "data" / "sfhq_t2i" / f, d / f)
    return tmp_path


def jpeg(d, name=SDXL, size=(1024, 1024), fmt="JPEG", color=(150, 110, 90)):
    p = d / name
    Image.new("RGB", size, color).save(p, fmt)
    return p


def meta_csv(tmp_path, name=SDXL, model="SDXL", engine="stable-diffusion-xl-1024-v1-0", preset="photographic"):
    p = tmp_path / "SFHQ_T2I_dataset.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_filename", "model_used", "text_prompt", "configs"])
        w.writerow([name, model, "a portrait", json.dumps({"engine_id": engine, "style_preset": preset, "seed": 1})])
    return p


def data(proj):
    return proj / "data" / "sfhq_t2i"


def cfgp(proj):
    return proj / "config" / "pilot.yaml"


def manifest_rows(proj):
    with open(data(proj) / "MANIFEST.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_register_pilot_without_csv(proj):
    img = jpeg(data(proj))
    r = register(img, "1 cara, frontal", pilot=True, config_path=cfgp(proj), today=date(2026, 10, 21))
    rows = manifest_rows(proj)
    assert len(rows) == 1 and rows[0]["image_filename"] == SDXL and rows[0]["sha256"] == sha256_file(img)
    assert rows[0]["model_used"] == "SDXL" and rows[0]["role"] == "pilot"
    assert rows[0]["style_preset"] == "no verificado (sin CSV)" and "prefijo del nombre" in rows[0]["source"]
    assert rows[0]["license"] == LICENSE_NOTE and "no es una verificación legal" in LICENSE_NOTE
    assert "registrada 2026-10-21" in rows[0]["source"]
    text = cfgp(proj).read_text(encoding="utf-8")
    assert yaml.safe_load(text)["dataset"]["pilot_image"] == SDXL
    assert "# solo el nombre de archivo" in text and text.count("\n") == (ROOT / "config" / "pilot.yaml").read_text(encoding="utf-8").count("\n")
    assert len(re.findall(r"^\s+pilot_image:", text, re.M)) == 1
    assert r["model_source"] == "prefijo del nombre de archivo"


def test_register_with_csv_records_verified_style(proj, tmp_path):
    jpeg(data(proj))
    register(data(proj) / SDXL, "ok", pilot=True, csv_path=meta_csv(tmp_path), config_path=cfgp(proj))
    row = manifest_rows(proj)[0]
    assert row["style_preset"] == "photographic" and "CSV oficial" in row["source"]


@pytest.mark.parametrize("kw,needle", [
    ({"model": "FLUX1_dev"}, "El CSV dice model_used=FLUX1_dev"),
    ({"engine": "otro-motor"}, "engine_id"),
    ({"preset": "enhance"}, "no es fotográfico"),
    ({"name": "SDXL_image_0009999.jpg"}, "no aparece en"),
])
def test_csv_inconsistencies_block(proj, tmp_path, kw, needle):
    jpeg(data(proj))
    with pytest.raises(RegistrationError, match=needle):
        register(data(proj) / SDXL, "ok", pilot=True, csv_path=meta_csv(tmp_path, **kw), config_path=cfgp(proj))
    assert manifest_rows(proj) == [] and yaml.safe_load(cfgp(proj).read_text(encoding="utf-8"))["dataset"]["pilot_image"] is None


@pytest.mark.parametrize("name,size,fmt,needle", [
    ("foto.jpg", (1024, 1024), "JPEG", "no sigue el patrón"),
    ("FLUX1_dev_image_0000001.jpg", (1024, 1024), "JPEG", "no permitido"),
    (SDXL, (512, 512), "JPEG", "tamaño"),
    (SDXL, (1024, 1024), "PNG", "formato PNG"),
])
def test_invalid_image_blocks(proj, name, size, fmt, needle):
    jpeg(data(proj), name, size, fmt)
    with pytest.raises(RegistrationError, match=needle):
        register(data(proj) / name, "ok", pilot=True, config_path=cfgp(proj))
    assert manifest_rows(proj) == []


def test_missing_reason_and_wrong_folder(proj, tmp_path):
    jpeg(data(proj))
    with pytest.raises(RegistrationError, match="Falta --motivo"):
        register(data(proj) / SDXL, "  ", config_path=cfgp(proj))
    outside = jpeg(tmp_path)
    with pytest.raises(RegistrationError, match="debe estar en"):
        register(outside, "ok", config_path=cfgp(proj))
    with pytest.raises(RegistrationError, match="No existe"):
        register(data(proj) / "SDXL_image_0000000.jpg", "ok", config_path=cfgp(proj))


def test_idempotent_and_changed_file(proj):
    img = jpeg(data(proj))
    register(img, "ok", pilot=True, config_path=cfgp(proj))
    assert register(img, "ok", pilot=True, config_path=cfgp(proj))["already_registered"]
    assert len(manifest_rows(proj)) == 1
    jpeg(data(proj), color=(10, 200, 10))  # mismo nombre, otro contenido
    with pytest.raises(RegistrationError, match="otro SHA-256"):
        register(img, "ok", pilot=True, config_path=cfgp(proj))


def test_second_pilot_blocked_reserve_allowed(proj):
    register(jpeg(data(proj)), "ok", pilot=True, config_path=cfgp(proj))
    other = jpeg(data(proj), "SDXL_image_0000002.jpg", color=(1, 2, 3))
    with pytest.raises(RegistrationError, match="Ya hay una imagen piloto"):
        register(other, "ok", pilot=True, config_path=cfgp(proj))
    register(other, "reserva para la demo", config_path=cfgp(proj))
    roles = {r["image_filename"]: r["role"] for r in manifest_rows(proj)}
    assert roles == {SDXL: "pilot", "SDXL_image_0000002.jpg": "reserva"}


def test_bad_manifest_header_blocks(proj):
    (data(proj) / "MANIFEST.csv").write_text("nombre,hash\n", encoding="utf-8")
    jpeg(data(proj))
    with pytest.raises(RegistrationError, match="Cabecera"):
        register(data(proj) / SDXL, "ok", config_path=cfgp(proj))


# --- coherencia de la carpeta y make_zip --------------------------------------

def test_model_from_name():
    assert model_from_name(SDXL) == "SDXL" and model_from_name("FLUX1_dev_image_0000001.jpg") == "FLUX1_dev"
    assert model_from_name("SDXL_image_12.jpg") is None


def test_check_dataset_dir_problems(proj):
    d = data(proj)
    assert check_dataset_dir(d, ["SDXL"]) == []
    jpeg(d)
    assert any("no registrada" in p for p in check_dataset_dir(d, ["SDXL"]))
    register(d / SDXL, "ok", pilot=True, config_path=cfgp(proj))
    assert check_dataset_dir(d, ["SDXL"], SDXL) == []
    (d / "SFHQ_T2I_dataset.csv").write_text("x", encoding="utf-8")  # CSV de metadatos olvidado en la carpeta
    assert any("archivo no registrada" in p and "SFHQ_T2I_dataset.csv" in p for p in check_dataset_dir(d, ["SDXL"]))
    (d / "SFHQ_T2I_dataset.csv").unlink()
    jpeg(d, color=(9, 9, 9))  # contenido cambiado tras registrar
    assert any("SHA-256" in p for p in check_dataset_dir(d, ["SDXL"]))
    (d / SDXL).unlink()
    assert any("no existe" in p for p in check_dataset_dir(d, ["SDXL"]))
    assert any("no está registrada" in p for p in check_dataset_dir(d, ["SDXL"], "SDXL_image_0000777.jpg"))


def test_make_zip_rejects_unregistered_image(proj):
    jpeg(data(proj))
    assert any("no registrada" in p for p in make_zip.data_problems(cfgp(proj)))
    register(data(proj) / SDXL, "ok", pilot=True, config_path=cfgp(proj))
    assert make_zip.data_problems(cfgp(proj)) == []


def test_real_project_data_is_coherent():
    assert make_zip.data_problems() == []


def test_manifest_header_constant_matches_file():
    with open(ROOT / "data" / "sfhq_t2i" / "MANIFEST.csv", newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == MANIFEST_FIELDS
