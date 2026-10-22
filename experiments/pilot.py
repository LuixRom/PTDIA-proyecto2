"""Piloto A1 vs A2 (planificacion/10 §3). Lo llama run_colab.ipynb; con backend='mock' se prueba en local.

Salida en out_dir: input_512.png, images/<método>/V<k>_s<seed>.png, runs.jsonl (incremental y reanudable),
metrics.csv, set_metrics.csv, contact_<método>.png, blind/ + blind_key.json, env.json, config.yaml,
LIMITATIONS.md, MANIFEST.sha256.
"""
import csv
import json
import random
import shutil
import time
from pathlib import Path

from PIL import Image

from p2 import env_check, export, generate, metrics
from p2.preprocess import count_faces, prepare
from p2.schemas import RunRecord

METHODS = ("A1", "A2")


def _load_runs(path):
    if not path.exists():
        return {}
    recs = [RunRecord.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(r.role, r.method, r.variation): r for r in recs}


def _vram_tracker():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.reset_peak_memory_stats, lambda: round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except ImportError:
        pass
    return (lambda: None), (lambda: None)


def _require_ok(validation, what):
    # Segunda barrera además de las celdas del notebook: sin validación correcta no hay modelos reales.
    if not (isinstance(validation, dict) and validation.get("ok") is True):
        raise env_check.ValidationError(f"{what} requiere una validación de compatibilidad correcta (celda 4).")


def generate_all(cfg, input_path, out_dir, backend="sd15", validation=None, strength=None, pipe=None, warmup=None):
    """strength: sustituye generation.strength (barrido del piloto 2); pipe: pipeline ya cargado (se reutiliza
    entre strengths); warmup: None = lo que diga la configuración."""
    if backend == "sd15":
        _require_ok(validation, "La generación con SD 1.5")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img, sha = prepare(Path(input_path).read_bytes(), cfg["image_size"])
    img.save(out / "input_512.png")
    gen = dict(cfg["generation"])
    if strength is not None:
        gen["strength"] = strength
    if backend == "sd15":
        pipe = pipe if pipe is not None else generate.load_sd15(cfg)
        one = lambda im, spec: generate.sd15_generate(pipe, im, spec, gen)  # noqa: E731
        model_id = f"{cfg['models']['sd15']['repo']}@{cfg['models']['sd15']['revision']}"
    elif backend == "mock":
        one, model_id = generate.mock_generate, "mock"
    else:
        raise ValueError(f"backend desconocido: {backend}")

    runs_path = out / "runs.jsonl"
    done = _load_runs(runs_path)
    reset_peak, peak = _vram_tracker()

    def run(role, spec, rel):
        key = (role, spec["method"], spec["id"])
        if key in done:  # reanudación tras una desconexión
            return done[key]
        reset_peak()
        t0 = time.perf_counter()
        im, nsfw = one(img, spec)
        dt = time.perf_counter() - t0
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        im.save(p)
        rec = RunRecord(run_id=cfg["run_name"], method=spec["method"], variation=spec["id"], attribute=spec["attribute"],
                        prompt=spec["prompt"], negative_prompt=gen["negative_prompt"], seed=spec["seed"],
                        steps=gen["steps"], strength=gen["strength"], guidance_scale=gen["guidance_scale"],
                        scheduler=gen["scheduler"], width=im.width, height=im.height, model=model_id,
                        input_sha256=sha, output=rel, output_sha256=export.sha256_file(p), time_s=round(dt, 3),
                        vram_peak_gb=peak(), nsfw=nsfw, role=role)
        with open(runs_path, "a", encoding="utf-8") as f:
            f.write(rec.model_dump_json() + "\n")
        done[key] = rec
        return rec

    if gen.get("warmup") if warmup is None else warmup:
        run("warmup", generate.plan(cfg, "A1")[0], "warmup/V1.png")
    for m in METHODS:
        specs = generate.plan(cfg, m)
        if len(specs) != 5:
            raise AssertionError("Cada método debe producir exactamente 5 variaciones.")
        for spec in specs:
            run("main", spec, f"images/{m}/{spec['id']}_s{spec['seed']}.png")
    dc = gen["determinism_check"]
    spec = generate.plan(cfg, dc["method"])[dc["k"] - 1]
    rep = run("determinism", spec, f"determinism/{spec['id']}.png")
    main = done[("main", dc["method"], spec["id"])]
    result = {"input_sha256": sha, "n_main": sum(1 for k in done if k[0] == "main"),
              "determinism_identical": rep.output_sha256 == main.output_sha256}
    (out / "determinism.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def compute_metrics(cfg, out_dir, models=None, validation=None):
    if models is not None:
        _require_ok(validation, "La evaluación con CLIP/LPIPS")
    out = Path(out_dir)
    orig = Image.open(out / "input_512.png").convert("RGB")
    runs = [r for r in _load_runs(out / "runs.jsonl").values() if r.role == "main"]
    texts = cfg["attribute_texts"]
    rows, sets = [], []
    for m in METHODS:
        rs = sorted((r for r in runs if r.method == m), key=lambda r: r.variation)
        ims = [Image.open(out / r.output).convert("RGB") for r in rs]
        for r, im in zip(rs, ims):
            k = int(r.variation[1:])
            row = {"method": m, "variation": r.variation, "attribute": r.attribute, "seed": r.seed, "nsfw": r.nsfw}
            row.update(metrics.per_image(orig, im, texts[k - 1], models, count_faces))
            rows.append(row)
        sets.append({"method": m, "n": len(ims), **metrics.per_set(ims, models),
                     "lpips_available": models is not None and models.lpips is not None})
    for name, data in (("metrics.csv", rows), ("set_metrics.csv", sets)):
        keys = list(dict.fromkeys(k for d in data for k in d))
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
    return rows, sets


def contact_sheets(out_dir, seed=2026):
    out = Path(out_dir)
    orig = Image.open(out / "input_512.png")
    for m in METHODS:
        files = sorted((out / "images" / m).glob("V*.png"))
        sheet = Image.new("RGB", (256 * 6, 256), "white")
        for i, p in enumerate([out / "input_512.png", *files]):
            sheet.paste(Image.open(p).resize((256, 256)), (256 * i, 0))
        sheet.save(out / f"contact_{m}.png")
    # Conjunto ciego: IDs aleatorios, sin método; la clave queda FUERA de blind/.
    items = [(m, p) for m in METHODS for p in sorted((out / "images" / m).glob("V*.png"))]
    rng = random.Random(seed)
    rng.shuffle(items)
    blind = out / "blind"
    blind.mkdir(exist_ok=True)
    orig.save(blind / "original.png")
    key = {}
    for i, (m, p) in enumerate(items, start=1):
        bid = f"B{i:02d}"
        shutil.copy(p, blind / f"{bid}.png")
        key[bid] = {"method": m, "file": p.relative_to(out).as_posix()}
    (out / "blind_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    return key


def sweep_summary(base_dir, strengths):
    """Una fila por (strength, método) con las métricas que responden a la pregunta del piloto 2:
    ¿a qué strength se aplica el atributo (CLIPΔ) sin perder el parecido (similitud CLIP)?"""
    def mean(xs):
        xs = [float(x) for x in xs if x not in ("", None)]
        return round(sum(xs) / len(xs), 4) if xs else None
    out = []
    for s in strengths:
        d = Path(base_dir) / f"strength_{s:.2f}"
        with open(d / "metrics.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with open(d / "set_metrics.csv", newline="", encoding="utf-8") as f:
            sets = {r["method"]: r for r in csv.DictReader(f)}
        for m in METHODS:
            rs = [r for r in rows if r["method"] == m]
            sims = [float(r["clip_img_sim"]) for r in rs if r.get("clip_img_sim")]
            out.append({"strength": s, "method": m, "n": len(rs),
                        "clip_delta_mean": mean(r.get("clip_delta") for r in rs),
                        "clip_img_sim_mean": mean(sims), "clip_img_sim_min": round(min(sims), 4) if sims else None,
                        "clip_div": sets[m].get("clip_div") or None, "ssim_div": sets[m].get("ssim_div") or None,
                        "faces_eq_1": sum(1 for r in rs if r.get("faces") == "1"),
                        "nsfw": sum(1 for r in rs if r.get("nsfw") == "True"),
                        "black": sum(1 for r in rs if r.get("black") == "True")})
    keys = list(out[0]) if out else []
    with open(Path(base_dir) / "sweep_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    return out


LIMITATIONS = """# Limitaciones de esta ejecución (completar tras la evaluación)
- Muestra: 1 imagen de entrada x 2 métodos x 5 variaciones x 1 semilla por strength. Resultados EXPLORATORIOS: no generalizables.
- Piloto 1 (strength 0.40) se ejecutó aparte; el piloto 2 barre los strengths de config generation.strengths.
- Las métricas automáticas no miden identidad; CLIP no se usa para reconocer personas.
- Sesgos del generador del dataset (SDXL) y de SD 1.5: anotar cambios no pedidos (tono de piel, edad, rasgos).
- Activaciones del safety checker: ver columna nsfw en metrics.csv.
- Si lpips_available = False en set_metrics.csv, no hay distancia perceptual LPIPS: solo SSIM, CLIP y juicio humano.
- Versiones exactas usadas (p.ej. diffusers 0.41.0 en vez del 0.40.0 probado): ver validation.json y env.json.
- Imagen de entrada sintética (SFHQ-T2I): parecido accidental con personas reales no descartable.
"""


def finalize(cfg, out_dir, config_path, validation_path=None):
    out = Path(out_dir)
    (out / "env.json").write_text(json.dumps(env_check.collect(str(out)), indent=2), encoding="utf-8")
    if validation_path and Path(validation_path).exists():
        shutil.copy(validation_path, out / "validation.json")
    shutil.copy(config_path, out / "config.yaml")
    (out / "LIMITATIONS.md").write_text(LIMITATIONS, encoding="utf-8")
    return export.write_manifest(out)
