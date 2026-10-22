"""Simula run_colab.ipynb en local, ejecutando sus celdas REALES en orden, antes de subirlo a Colab.

Uso (con el entorno que tiene torch/diffusers/transformers):
    .venv\\Scripts\\python.exe tools\\simulate_colab.py

Qué se sustituye SOLO en la simulación (el notebook no se modifica):
- google.colab (subir/descargar/Drive) por un stub; "/content" por una carpeta temporal.
- SD 1.5 y CLIP por modelos DIMINUTOS con pesos aleatorios creados aquí, con las mismas clases de
  diffusers/transformers (StableDiffusionPipeline + safety checker, CLIPModel en .bin). No se descarga nada.
  Las imágenes resultantes son ruido: solo se comprueba que el flujo y las APIs funcionan.
- Umbrales de recursos de la T4 (12 GB VRAM) relajados, porque la GPU local tiene 4 GB.
Escenarios: A) por defecto (sin autorizaciones), B) piloto autorizado completo, C) celdas fuera de orden,
D) código desde ZIP subido. Termina con código 0 solo si todos se comportan como se espera.
"""
import base64
import contextlib
import io
import json
import shutil
import sys
import tempfile
import traceback
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- modelos diminutos (sin descargas)

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


def tiny_tokenizer(d):
    from transformers import CLIPTokenizer
    chars = _bytes_to_unicode()
    vocab = {t: i for i, t in enumerate(["<|startoftext|>", "<|endoftext|>"] + chars + [c + "</w>" for c in chars])}
    d.mkdir(parents=True, exist_ok=True)
    (d / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (d / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    return CLIPTokenizer(str(d / "vocab.json"), str(d / "merges.txt"), model_max_length=77), len(vocab)


def install_old_torchao(root):
    """Imita el torchao preinstalado en Colab (oct-2026): existe, pero sin torchao.quantization.FqnToConfig."""
    d = root / "fake_site"
    (d / "torchao" / "quantization").mkdir(parents=True, exist_ok=True)
    (d / "torchao" / "__init__.py").write_text('__version__ = "0.10.0"\n', encoding="utf-8")
    (d / "torchao" / "quantization" / "__init__.py").write_text("def quantize_(*a, **k):\n    pass\n", encoding="utf-8")
    di = d / "torchao-0.10.0.dist-info"
    di.mkdir(exist_ok=True)
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: torchao\nVersion: 0.10.0\n", encoding="utf-8")
    (di / "top_level.txt").write_text("torchao\n", encoding="utf-8")
    sys.path.insert(0, str(d))
    return d


def build_tiny_models(root):
    import torch
    from p2.compat import neutralize_incompatible_torchao
    print("Compatibilidad:", neutralize_incompatible_torchao())
    from diffusers import AutoencoderKL, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTextConfig, CLIPTextModel

    torch.manual_seed(0)
    tok, nv = tiny_tokenizer(root / "_tok")
    txt = dict(vocab_size=nv, hidden_size=32, intermediate_size=37, num_hidden_layers=2, num_attention_heads=4,
               max_position_embeddings=77, bos_token_id=0, eos_token_id=1, pad_token_id=1)
    sd = StableDiffusionPipeline(
        vae=AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, layers_per_block=1,
                          block_out_channels=(32, 32, 32, 32), down_block_types=["DownEncoderBlock2D"] * 4,
                          up_block_types=["UpDecoderBlock2D"] * 4, sample_size=512),
        text_encoder=CLIPTextModel(CLIPTextConfig(**txt, projection_dim=32)),
        tokenizer=tok,
        unet=UNet2DConditionModel(sample_size=64, in_channels=4, out_channels=4, layers_per_block=1,
                                  block_out_channels=(32, 64), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                                  up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=32,
                                  attention_head_dim=8),
        scheduler=PNDMScheduler(skip_prk_steps=True),
        safety_checker=StableDiffusionSafetyChecker(CLIPConfig(
            text_config=txt, vision_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=2,
                                                num_attention_heads=4, image_size=32, patch_size=4), projection_dim=32)),
        feature_extractor=CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}),
        image_encoder=None, requires_safety_checker=True)
    sd_dir = root / "tiny-sd15"
    sd.save_pretrained(sd_dir, safe_serialization=True, variant="fp16")  # como el repo real: *.fp16.safetensors

    clip = CLIPModel(CLIPConfig(text_config=txt, vision_config=dict(hidden_size=32, intermediate_size=37,
                                num_hidden_layers=2, num_attention_heads=4, image_size=224, patch_size=32),
                                projection_dim=32))
    clip_dir = root / "tiny-clip"
    clip.save_pretrained(clip_dir, safe_serialization=False)  # como el repo real: pytorch_model.bin
    CLIPProcessor(image_processor=CLIPImageProcessor(size={"shortest_edge": 224},
                                                     crop_size={"height": 224, "width": 224}),
                  tokenizer=tok).save_pretrained(clip_dir)
    files = sorted(p.relative_to(root).as_posix() for p in (sd_dir.rglob("*.safetensors"))) + \
        sorted(p.relative_to(root).as_posix() for p in clip_dir.glob("*.bin"))
    from transformers import Qwen2Config, Qwen2ForCausalLM
    qtok, _ = tiny_tokenizer(root / "_qtok")  # mismo vocabulario byte-level; se añaden los tokens de chat de Qwen
    qtok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    qtok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    qwen = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(qtok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                        bos_token_id=0, eos_token_id=1, pad_token_id=1))
    qwen_dir = root / "tiny-qwen"
    qwen.save_pretrained(qwen_dir)
    qtok.save_pretrained(qwen_dir)
    return sd_dir, clip_dir, qwen_dir, files


# ---------------------------------------------------------------- stubs de Colab / IPython

class Colab:
    def __init__(self, content, upload_file=None):
        self.content, self.upload_file, self.downloads, self.mounted = content, upload_file, [], []

    def install(self):
        files = types.SimpleNamespace(upload=self.upload, download=self.downloads.append)
        drive = types.SimpleNamespace(mount=self.mount)
        colab = types.ModuleType("google.colab")
        colab.files, colab.drive = files, drive
        google = sys.modules.get("google") or types.ModuleType("google")
        google.colab = colab
        sys.modules.update({"google": google, "google.colab": colab})
        try:
            import IPython.display  # noqa: F401
        except ImportError:
            ipy, disp = types.ModuleType("IPython"), types.ModuleType("IPython.display")
            disp.display = lambda x: print(x)
            ipy.display = disp
            sys.modules.update({"IPython": ipy, "IPython.display": disp})

    def upload(self):
        if self.upload_file is None:
            return {}
        dst = self.content / self.upload_file.name
        shutil.copy(self.upload_file, dst)
        return {dst.name: dst.read_bytes()}

    def mount(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.mounted.append(path)


# ---------------------------------------------------------------- ejecución de celdas

def code_cells(nb_path, content):
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    cells = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"]).replace("/content", content.as_posix())
        cells.append((src.splitlines()[0].replace("#@title ", ""), src))
    return cells


def run(cells, ns, hooks=None, quiet=True):
    """Ejecuta celdas en orden. Devuelve [(título, 'ok'|'detenida'|'ERROR', mensaje)] y se para en la primera no-ok."""
    out = []
    for title, src in cells:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
                exec(compile(src, title, "exec"), ns)
            out.append((title, "ok", buf.getvalue().strip().splitlines()[-1:] or [""]))
            if hooks and title in hooks:
                hooks[title](ns)
        except SystemExit as e:
            out.append((title, "detenida", [str(e)]))
            break
        except Exception:
            out.append((title, "ERROR", traceback.format_exc().splitlines()[-6:]))
            break
    return out


def fresh_ns():
    return {"__name__": "__main__"}


def relax_resources(ns):
    import p2.env_check as ec
    if getattr(ec.problems, "_sim", False):
        return
    orig = ec.problems

    def relaxed(info, req):
        return orig(info, {**req, "min_vram_gb": 3, "min_ram_gb": 1, "min_disk_gb": 5})
    relaxed._sim = True
    ec.problems = relaxed


def report(name, res, expect):
    print(f"\n=== Escenario {name}")
    for title, status, msg in res:
        print(f"  [{status:8}] {title}")
        for m in msg:
            if m:
                print(f"             {m[:300]}")
    ok = expect(res)
    print(f"  => {'COMO SE ESPERABA' if ok else 'NO ESPERADO'}")
    return ok


def main():
    from tools.make_zip import main as make_zip_main

    nb = ROOT / "run_colab.ipynb"
    zip_path = make_zip_main()  # antes de ejecutar celdas: la celda 2 cambia sys.path y sys.modules
    tmp = Path(tempfile.mkdtemp(prefix="p2_colab_sim_"))
    print("Carpeta de simulación:", tmp)
    content = tmp / "content"
    content.mkdir()
    install_old_torchao(tmp)  # como en Colab: torchao instalado pero sin FqnToConfig
    sd_dir, clip_dir, qwen_dir, wfiles = build_tiny_models(tmp / "models")
    print("Modelos diminutos creados en local:", wfiles)
    results = []

    # A) por defecto: sin autorizaciones -> 1..5 ok y la 6 se detiene pidiendo autorización
    colab = Colab(content)
    colab.install()
    cells = code_cells(nb, content)
    ns = fresh_ns()
    resA = run(cells, ns, hooks={"2. Preparar el código del proyecto y verificar su integridad": relax_resources})
    # «Ejecutar todas» por defecto: ninguna excepción; 6-8 se omiten con aviso.
    results.append(report("A (por defecto, sin autorizaciones: Ejecutar todas)", resA, lambda r: (
        len(r) == len(cells) and all(s == "ok" for _, s, _ in r)
        and any("Piloto NO ejecutado" in m for t, _, ms in r if t.startswith("6.") for m in ms)
        and all("omitida" in " ".join(ms) for t, _, ms in r if t.startswith(("7.", "8."))))))

    # B) piloto autorizado completo con modelos diminutos
    def authorize(n):
        n["AUTORIZO_DESCARGAR_MODELOS"] = n["AUTORIZO_EJECUTAR_PILOTO"] = n["PILOTO_AUTORIZADO"] = True

    def tiny(n):
        n["cfg"]["models"]["sd15"]["repo"] = str(sd_dir)
        n["cfg"]["models"]["clip"]["repo"] = str(clip_dir)
        n["cfg"]["llm"]["repo"] = str(qwen_dir)
    ns = fresh_ns()
    resB = run(cells, ns, hooks={"1. Parámetros y autorizaciones": authorize,
                                 "2. Preparar el código del proyecto y verificar su integridad": relax_resources,
                                 "3. Comprobar GPU, VRAM, RAM, disco y versiones (no instala nada)": tiny})
    okB = report("B (piloto autorizado, modelos diminutos)", resB, lambda r: all(s == "ok" for _, s, _ in r) and len(r) == len(cells))
    if okB:
        out, strengths = ns["OUT"], ns["STRENGTHS"]
        checks = {f"strengths del piloto 2 = {strengths}": strengths == [0.55, 0.70]}
        allruns = []
        for i, s in enumerate(strengths):
            d = out / f"strength_{s:.2f}"
            runs = [json.loads(l) for l in (d / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
            allruns += runs
            roles = {k: sum(1 for r in runs if r["role"] == k) for k in ("warmup", "main", "determinism")}
            checks[f"s={s}: {'1 calentamiento + ' if i == 0 else ''}10 + 1 determinismo"] =                 roles == {"warmup": 1 if i == 0 else 0, "main": 10, "determinism": 1}
            checks[f"s={s}: strength registrado en runs.jsonl"] = all(r["strength"] == s for r in runs)
            checks[f"s={s}: A1 y A2 con semillas 1001-1005"] = all(
                sorted(r["seed"] for r in runs if r["role"] == "main" and r["method"] == m) == [1001, 1002, 1003, 1004, 1005]
                for m in ("A1", "A2"))
            checks[f"s={s}: determinismo"] = json.loads((d / "determinism.json").read_text())["determinism_identical"]
            checks[f"s={s}: conjunto ciego de 10 + clave aparte"] = len(list((d / "blind").glob("B*.png"))) == 10                 and (d / "blind_key.json").exists()
        checks["sweep_summary.csv: 2 strengths x 2 métodos"] = len(ns["summary"]) == 4
        checks["ZIP de resultados descargado"] = bool(colab.downloads) and Path(colab.downloads[-1]).exists()
        from p2.export import verify_manifest
        checks["manifiesto de resultados íntegro"] = verify_manifest(out) == []
        with zipfile.ZipFile(colab.downloads[-1]) as z:
            names = set(z.namelist())
            checks["ZIP contiene validation.json, sweep_summary.csv y ambas carpetas"] = {
                "validation.json", "sweep_summary.csv", "strength_0.55/runs.jsonl", "strength_0.70/runs.jsonl"} <= names
        for k, v in checks.items():
            print(f"  {'✓' if v else '✗'} {k}")
        print(f"  Generaciones totales: {len(allruns)} | NSFW (modelo aleatorio): {sum(1 for r in allruns if r['nsfw'])}")
        okB = all(checks.values())
    results.append(okB)

    # C) fuera de orden: celda 7 sin las anteriores, y celda 3 sin la 1
    by = {t.split(".")[0]: (t, s) for t, s in cells}
    resC1 = run([by["7"]], fresh_ns())
    resC2 = run([by["1"], by["7"]], fresh_ns(), hooks={by["1"][0]: authorize})
    results.append(report("C1 (celda 7 sin ejecutar nada antes)", resC1,
                          lambda r: r[0][1] == "detenida" and "Ejecute las celdas en orden" in r[0][2][0]))
    results.append(report("C2 (celdas 1 y 7, saltando 2-6)", resC2,
                          lambda r: r[-1][1] == "detenida" and "Falta ejecutar celdas anteriores" in r[-1][2][0]))

    # D) código desde ZIP subido (alternativa a "incluido")
    colab_d = Colab(content, upload_file=zip_path)
    colab_d.install()

    def use_upload(n):
        n["FUENTE_CODIGO"] = "subir"
    nsD = fresh_ns()
    resD = run(cells[:4], nsD, hooks={"1. Parámetros y autorizaciones": use_upload,
                                      "2. Preparar el código del proyecto y verificar su integridad": relax_resources})
    results.append(report("D (FUENTE_CODIGO = subir, ZIP de dist/)", resD, lambda r: all(s == "ok" for _, s, _ in r)
                          and nsD.get("zip_path") is not None and nsD["zip_path"].name == zip_path.name))

    # E) App final (celdas 9-10) con SD y Qwen2 diminutos. Sin gradio en el .venv: se sustituye solo la ventana
    #    (build_ui) por un recorrido que usa las funciones REALES de la app: sesión, 5 decisiones, resumen LLM, verificación.
    app_run = {}

    def authorize_app(n):
        n["AUTORIZO_DESCARGAR_MODELOS"] = n["AUTORIZO_APP"] = n["APP_AUTORIZADA"] = True
        n["PILOTO_AUTORIZADO"] = False
        sys.modules.setdefault("gradio", types.ModuleType("gradio"))  # la celda 9 solo comprueba que existe

    def fake_window(n):
        relax_resources(n)
        import p2.ui as ui

        def build_ui(store, cfg, method, generate_one, out_dir, llm, llm_name, gen, db=None):
            class Demo:
                def close(self):  # como gradio.Blocks.close (lo usa la celda 11)
                    app_run["closed"] = True

                def launch(self, **kw):
                    img = Path(cfg["dataset"]["dir"]) / cfg["dataset"]["pilot_image"]
                    sid, paths = ui.new_session(store, img.read_bytes(), cfg, method, generate_one, out_dir, gen, "alias-1")
                    decs = ["aceptada", "rechazada", "aceptada", "rechazada", "rechazada"]
                    for k, d in enumerate(decs, start=1):
                        ui.on_decide(store, sid, f"V{k}", d, f"motivo {k}")
                    res = ui.on_summarize(store, sid, llm, llm_name, db)
                    score = ui.on_sus(store, sid, "alias-1", [4, 2, 4, 2, 4, 2, 4, 2, 4, 2], db)
                    app_run.update(sid=sid, launch=kw, res=res, store=store, db=db, score=score)
            return Demo()
        ui.build_ui = build_ui

    colab_e = Colab(content)
    colab_e.install()
    nsE = fresh_ns()
    resE = run(cells, nsE, hooks={"1. Parámetros y autorizaciones": authorize_app,
                                  "2. Preparar el código del proyecto y verificar su integridad": fake_window,
                                  "3. Comprobar GPU, VRAM, RAM, disco y versiones (no instala nada)": tiny})
    okE = report("E (app final: celdas 9-10, SD + Qwen2 diminutos)", resE,
                 lambda r: len(r) == len(cells) and all(s == "ok" for _, s, _ in r))
    if okE:
        store, sid, res = app_run["store"], app_run["sid"], app_run["res"]
        s = store.load(sid)
        summ = json.loads((store.root / sid / "summary.json").read_text(encoding="utf-8"))
        checks = {
            "la app se lanza dentro de la celda (share=False, inline=True)":
                app_run["launch"].get("share") is False and app_run["launch"].get("inline") is True,
            "launch autoriza la carpeta de sesiones (allowed_paths) y muestra errores":
                str(store.root) in app_run["launch"].get("allowed_paths", []) and app_run["launch"].get("show_error") is True,
            "5 variaciones A2 con strength 0.55 registrado": len(s.variations) == 5 and all(
                v.method == "A2" and v.params["strength"] == 0.55 for v in s.variations),
            "decisiones registradas (2 aceptadas, 3 rechazadas)": s.accepted() == ["V1", "V3"] and s.rejected() == ["V2", "V4", "V5"],
            "el LLM (Qwen2 diminuto) se llamó y el resultado quedó verificado": summ["verified"] is True
                and "tiny-qwen" in summ["llm"] and summ["attempts"] >= 1,
            "celda 10: ZIP de sesiones descargado": bool(colab_e.downloads) and "sesiones_app_" in colab_e.downloads[-1],
            "BD temporal: 1 respuesta del agente con alias y textos del LLM": [
                (r["session_id"], r["participante"]) for r in app_run["db"].rows("resumenes")] == [(sid, "alias-1")]
                and app_run["db"].rows("resumenes")[0]["respuestas_llm"].startswith("["),
            "BD temporal: SUS guardado (puntuación 75)": app_run["score"] == 75.0
                and [r["puntuacion"] for r in app_run["db"].rows("sus")] == [75.0],
            "el ZIP incluye la BD temporal y sus.jsonl": {"agente_tmp.sqlite", "sus.jsonl"} <= set(
                zipfile.ZipFile(colab_e.downloads[-1]).namelist()),
        }
        from p2.export import verify_manifest
        checks["manifiesto de sesiones íntegro"] = verify_manifest(store.root) == []
        for k, v in checks.items():
            print(f"  {'✓' if v else '✗'} {k}")
        print(f"  Resumen ({'plantilla de respaldo' if summ['fallback'] else 'LLM'}): {summ['summary']['resumen'][:160]}")
        okE = all(checks.values())
    results.append(okE)

    # F) Borrado de datos temporales (celda 11): sin confirmar no borra nada; confirmado borra sesiones, BD y ZIP.
    cell11 = [c for c in cells if c[0].startswith("11.")]
    resF1 = run(cell11, nsE)
    kept = (content / "sesiones").exists()
    nsE["CONFIRMO_BORRAR_DATOS"] = True
    resF2 = run(cell11, nsE)
    gone = not (content / "sesiones").exists() and not list(content.glob("sesiones_app_*.zip"))
    results.append(report("F (celda 11: borrado de datos temporales)", resF1 + resF2,
                          lambda r: all(s == "ok" for _, s, _ in r) and "No se borró nada" in " ".join(r[0][2])
                          and kept and gone))

    print("\nRESULTADO:", "TODO COMO SE ESPERABA" if all(results) else "HAY ESCENARIOS NO ESPERADOS")
    print("Carpeta de simulación (puede borrarse):", tmp)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
