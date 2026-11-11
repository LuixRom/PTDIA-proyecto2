"""Descarga, una sola vez, los modelos del probador para usarlos sin internet (respaldo local o tools/eval_local.py).

Uso:  .venv\\Scripts\\python.exe tools\\download_models.py [repo_extra ...]   (candidatos a comparar)

Pensado para redes con antivirus o proxy que inspeccionan HTTPS (error «self-signed certificate»): usa los
certificados de Windows (truststore), descarga un archivo a la vez por HTTP normal (sin xet) y reintenta.
Solo baja la configuración y los pesos fp16 en safetensors de cada componente (o los normales si no hay fp16).
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")              # el backend xet cortaba conexiones (WinError 10038)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    import truststore
except ImportError:  # pip trae su propia copia
    import pip._vendor.truststore as truststore
truststore.inject_into_ssl()  # certificados del sistema: confía en el certificado del antivirus/proxy como el navegador

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def wanted(files):
    """Configuración/tokenizer + por cada carpeta los safetensors fp16 (o los normales si no hay fp16)."""
    keep = [f for f in files if f.endswith((".json", ".txt", ".model")) and not f.startswith(".")]
    st = [f for f in files if f.endswith(".safetensors")]
    for f in st:
        folder = f.rsplit("/", 1)[0] if "/" in f else ""
        has_fp16 = any(g.endswith(".fp16.safetensors") and (g.rsplit("/", 1)[0] if "/" in g else "") == folder for g in st)
        if f.endswith(".fp16.safetensors") or not has_fp16:
            keep.append(f)
    if not st:  # repos antiguos (p. ej. la revisión fijada de CLIP) solo publican pytorch_model.bin
        keep += [f for f in files if f.endswith("pytorch_model.bin")]
    return keep


def main():
    from huggingface_hub import hf_hub_download, list_repo_files

    from p2 import config
    cfg = config.load("config/vitrina.yaml")
    m = cfg["models"]
    repos = [m["sd15_inpaint"], cfg["approaches"]["B"]["controlnet"], m["segmenter"], m["pose"], m["clip"]]
    extra = [a for a in sys.argv[1:] if a != "--llm"]
    if "--llm" in sys.argv:  # LLM explicador para la app local (tools/app_local.py), con su revisión fijada
        repos.append(cfg["llms"]["qwen"])
    repos += [{"repo": r, "revision": "main"} for r in extra]  # candidatos extra a comparar (repo de HF)
    for e in repos:
        repo, rev = e["repo"], e["revision"]
        files = wanted(list_repo_files(repo, revision=rev))
        print(f"\n{repo}: {len(files)} archivos")
        for f in files:
            for attempt in range(1, 6):
                try:
                    hf_hub_download(repo, f, revision=rev)
                    print("  ok", f)
                    break
                except Exception as ex:  # noqa: BLE001  (red inestable: se reintenta)
                    print(f"  reintento {attempt}/5 {f}: {type(ex).__name__}")
                    time.sleep(3 * attempt)
            else:
                raise SystemExit(f"No se pudo descargar {repo}/{f}. Revise la conexión y vuelva a ejecutar (se reanuda).")
    print("\nListo: modelos descargados. Ya se pueden usar sin internet (HF_HUB_OFFLINE=1).")


if __name__ == "__main__":
    main()
