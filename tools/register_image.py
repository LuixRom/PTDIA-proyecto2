"""Registra una imagen ya descargada en data/sfhq_t2i: valida, añade la fila a MANIFEST.csv y,
con --pilot, fija dataset.pilot_image en config/pilot.yaml. No descarga nada.

Uso:
  python tools/register_image.py data/sfhq_t2i/SDXL_image_0001234.jpg --pilot --motivo "1 cara, frontal, sin oclusión"
  (opcional) --csv C:/PTDIA/sfhq_meta/SFHQ_T2I_dataset.csv   verifica model_used, motor y estilo en los metadatos oficiales

Sin --csv, el modelo se toma del prefijo del nombre (así nombra los archivos el script del autor) y el estilo
queda como "no verificado (sin CSV)". No se inventa ningún dato.
"""
import argparse
import csv
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

from p2.dataset import (MANIFEST_FIELDS, PHOTO_PRESETS, SDXL_ENGINE, check_dataset_dir, image_problems,  # noqa: E402
                        model_from_name, read_manifest)
from p2.export import sha256_file  # noqa: E402

KAGGLE_REF = "selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models"
LICENSE_NOTE = ("MIT según el campo de licencia de la ficha de Kaggle (consultado 2026-10-21); "
                "aviso en LICENSE_MIT.txt; no es una verificación legal")


class RegistrationError(ValueError):
    pass


def csv_metadata(csv_path, name):
    """Fila oficial del CSV para `name` (model_used, engine, preset). Error claro si falta."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        row = next((r for r in csv.DictReader(f) if r.get("image_filename") == name), None)
    if row is None:
        raise RegistrationError(f"{name} no aparece en {Path(csv_path).name} (columna image_filename).")
    missing = [k for k in ("model_used", "configs") if not row.get(k)]
    if missing:
        raise RegistrationError(f"La fila de {name} en el CSV no tiene {missing}.")
    try:
        cfg = json.loads(row["configs"])
    except json.JSONDecodeError as e:
        raise RegistrationError(f"La columna configs de {name} no es JSON válido ({e}).")
    return {"model_used": row["model_used"], "engine_id": cfg.get("engine_id"), "style_preset": cfg.get("style_preset")}


def set_pilot_image(config_path, name):
    """Reemplaza solo el valor de dataset.pilot_image, conservando comentarios y el resto del archivo."""
    text = Path(config_path).read_text(encoding="utf-8")
    pattern = re.compile(r"^(\s+pilot_image:\s*)(\S+)", re.M)
    if len(pattern.findall(text)) != 1:
        raise RegistrationError(f"No se encontró exactamente una línea 'pilot_image:' en {config_path}.")
    new = pattern.sub(lambda m: f'{m.group(1)}"{name}"', text)
    if yaml.safe_load(new)["dataset"]["pilot_image"] != name:
        raise RegistrationError("La edición de la configuración no produjo el valor esperado; no se guarda.")
    tmp = Path(str(config_path) + ".tmp")
    tmp.write_text(new, encoding="utf-8")
    os.replace(tmp, config_path)


def register(image_path, reason, pilot=False, csv_path=None, config_path=ROOT / "config" / "pilot.yaml", today=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_dir = (Path(config_path).parent.parent / cfg["dataset"]["dir"]).resolve()
    allowed = cfg["dataset"]["allowed_models"]
    img = Path(image_path).resolve()
    name = img.name

    errors = []
    if not reason or not reason.strip():
        errors.append("Falta --motivo: describa por qué se eligió la imagen (no se inventa).")
    if not img.exists():
        raise RegistrationError(f"No existe {img}. Descargue primero esa imagen a {data_dir}.")
    if img.parent != data_dir:
        errors.append(f"La imagen debe estar en {data_dir}, no en {img.parent}.")
    model = model_from_name(name)
    if model is None:
        errors.append(f"Nombre '{name}' no sigue el patrón SFHQ-T2I <MODELO>_image_NNNNNNN.jpg.")
    elif model not in allowed:
        errors.append(f"Modelo '{model}' (del nombre) no permitido; config dataset.allowed_models = {allowed}.")
    errors += image_problems(img)

    style = "no verificado (sin CSV)"
    model_source = "prefijo del nombre de archivo"
    if csv_path:
        meta = csv_metadata(csv_path, name)
        model_source = "CSV oficial SFHQ_T2I_dataset.csv"
        style = meta["style_preset"] or "ausente en el CSV"
        if model is not None and meta["model_used"] != model:
            errors.append(f"El CSV dice model_used={meta['model_used']}, el nombre dice {model}.")
        if meta["model_used"] == "SDXL":
            if meta["engine_id"] != SDXL_ENGINE:
                errors.append(f"engine_id={meta['engine_id']!r} en el CSV; se esperaba {SDXL_ENGINE}.")
            if meta["style_preset"] not in PHOTO_PRESETS:
                errors.append(f"style_preset={meta['style_preset']!r} no es fotográfico {sorted(PHOTO_PRESETS)}.")

    manifest = data_dir / "MANIFEST.csv"
    rows, mprobs = read_manifest(manifest)
    errors += mprobs
    digest = sha256_file(img)
    existing = next((r for r in rows if r["image_filename"] == name), None)
    if existing and existing["sha256"] != digest:
        errors.append(f"{name} ya está registrada con otro SHA-256: el archivo cambió.")
    other_pilots = [r["image_filename"] for r in rows if r["role"] == "pilot" and r["image_filename"] != name]
    if pilot and existing and existing["role"] != "pilot":
        errors.append(f"{name} ya está registrada con role={existing['role']}; cambie role a 'pilot' en MANIFEST.csv.")
    if pilot and other_pilots:
        errors.append(f"Ya hay una imagen piloto registrada: {other_pilots}. Edite MANIFEST.csv si quiere cambiarla.")
    if errors:
        raise RegistrationError("No se registró nada:\n- " + "\n- ".join(errors))

    if existing is None:
        row = {"image_filename": name, "sha256": digest, "model_used": model, "style_preset": style,
               "role": "pilot" if pilot else "reserva", "selection_reason": reason.strip(),
               "source": (f"Kaggle {KAGGLE_REF} v1, a_small_sample_new/{name}; modelo según {model_source}; "
                          f"registrada {(today or date.today()).isoformat()}"),
               "license": LICENSE_NOTE}
        with open(manifest, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writerow(row)
    if pilot:
        set_pilot_image(config_path, name)
    probs = check_dataset_dir(data_dir, allowed, name if pilot else cfg["dataset"]["pilot_image"])
    if probs:
        raise RegistrationError("Registro hecho, pero la carpeta no es coherente:\n- " + "\n- ".join(probs))
    return {"name": name, "sha256": digest, "already_registered": existing is not None, "pilot": pilot,
            "model_source": model_source, "style_preset": style}


def main():
    sys.stdout.reconfigure(errors="replace")  # consolas Windows cp1252: no fallar al imprimir «→» o «…»
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--motivo", required=True, help="por qué se eligió (1 cara, frontal, sin oclusión, ...)")
    ap.add_argument("--pilot", action="store_true", help="usarla como imagen del piloto (fija dataset.pilot_image)")
    ap.add_argument("--csv", help="SFHQ_T2I_dataset.csv oficial para verificar modelo, motor y estilo")
    a = ap.parse_args()
    try:
        r = register(a.image, a.motivo, a.pilot, a.csv)
    except RegistrationError as e:
        raise SystemExit(str(e))
    print(f"Registrada {r['name']} (sha256 {r['sha256'][:12]}…, modelo según {r['model_source']}, estilo: {r['style_preset']})"
          + (" — ya estaba registrada" if r["already_registered"] else ""))
    if r["pilot"]:
        print(f"config/pilot.yaml → dataset.pilot_image = {r['name']}. Siguiente: python tools/make_zip.py")


if __name__ == "__main__":
    main()
