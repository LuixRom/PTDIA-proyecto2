"""Comprobación de recursos y validación LOCAL de compatibilidad (sin red, sin pip, sin instalar nada).

Reglas del proyecto:
- Nunca se cambia (ni sube ni baja) un paquete instalado.
- La compatibilidad se verifica con los metadatos de los paquetes realmente instalados.
- Lo que no se puede verificar se informa como incertidumbre y BLOQUEA el paso siguiente.
- Las celdas que usan modelos reales exigen un estado de validación correcto (`require_validation`).
"""
import hashlib
import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .schemas import now_iso

KEY_PACKAGES = ["torch", "torchvision", "transformers", "accelerate", "huggingface-hub", "diffusers",
                "lpips", "safetensors", "tokenizers", "numpy", "pillow", "scikit-image", "opencv-python",
                "psutil", "pydantic", "pyyaml", "scipy", "tqdm", "packaging"]


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def installed_versions(names=KEY_PACKAGES, version_fn=_version):
    return {n: version_fn(n) for n in names}


def collect(disk_path="."):
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "disk_free_gb": round(shutil.disk_usage(disk_path).free / 1e9, 2),
            "ram_total_gb": None, "ram_available_gb": None,
            "cuda": False, "gpu": None, "vram_total_gb": None, "vram_free_gb": None,
            "packages": installed_versions()}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"], info["ram_available_gb"] = round(vm.total / 1e9, 2), round(vm.available / 1e9, 2)
    except ImportError:
        pass
    try:
        import torch
        info["torch_cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info.update(cuda=True, gpu=torch.cuda.get_device_name(0),
                        vram_total_gb=round(total / 1e9, 2), vram_free_gb=round(free / 1e9, 2))
    except ImportError:
        pass
    return info


def problems(info, req):
    """Lista de problemas de recursos bloqueantes, cada uno con la acción a tomar."""
    p = []
    if info["packages"].get("torch") is None:
        p.append("PyTorch no está instalado. Este notebook NO instala torch: use un entorno de Colab estándar.")
    if not info["cuda"]:
        p.append("No hay GPU CUDA. En Colab: Entorno de ejecución → Cambiar tipo de entorno → GPU T4.")
    elif info["vram_total_gb"] < req["min_vram_gb"]:
        p.append(f"VRAM total {info['vram_total_gb']} GB < {req['min_vram_gb']} GB requeridos. Use una T4 o superior.")
    if info["ram_total_gb"] is not None and info["ram_total_gb"] < req["min_ram_gb"]:
        p.append(f"RAM total {info['ram_total_gb']} GB < {req['min_ram_gb']} GB requeridos.")
    if info["disk_free_gb"] < req["min_disk_gb"]:
        p.append(f"Disco libre {info['disk_free_gb']} GB < {req['min_disk_gb']} GB requeridos. Libere espacio o reinicie el entorno.")
    return p


def _in_spec(version, spec, who):
    """(ok, problema, incertidumbre) para `version` frente al especificador `spec`."""
    try:
        return Version(version) in SpecifierSet(spec or "", prereleases=True), None
    except (InvalidVersion, InvalidSpecifier) as e:
        return False, f"{who}: no se puede interpretar la versión '{version}' o el requisito '{spec}' ({e})."


def check_declared_requirements(dist, version_fn=_version, requires_fn=metadata.requires):
    """Verifica, con los metadatos instalados de `dist`, que cada dependencia declarada está instalada
    y en rango. Devuelve (problemas, incertidumbres, comprobadas)."""
    probs, unc, checked = [], [], []
    try:
        reqs = requires_fn(dist) or []
    except metadata.PackageNotFoundError:
        return [f"{dist} no está instalado."], [], []
    except Exception as e:  # metadatos ilegibles
        return [], [f"{dist}: no se pudieron leer sus metadatos de dependencias ({e})."], []
    for raw in reqs:
        try:
            r = Requirement(raw)
        except InvalidRequirement as e:
            unc.append(f"{dist}: requisito no interpretable '{raw}' ({e}).")
            continue
        try:
            if r.marker is not None and not r.marker.evaluate({"extra": ""}):
                continue  # extra opcional o no aplica a esta plataforma
        except Exception as e:
            unc.append(f"{dist}: no se pudo evaluar el marcador de '{raw}' ({e}).")
            continue
        v = version_fn(r.name)
        if v is None:
            probs.append(f"{dist} requiere {r.name}{r.specifier}, que no está instalado.")
            continue
        ok, err = _in_spec(v, str(r.specifier), dist)
        if err:
            unc.append(err)
        elif not ok:
            probs.append(f"{dist} requiere {r.name}{r.specifier}; instalado {v}.")
        else:
            checked.append(f"{dist} → {r.name}{r.specifier or ''} ({v})")
    return probs, unc, checked


def fingerprint(versions):
    return hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()


def validate(cfg, version_fn=_version, requires_fn=metadata.requires):
    """Validación local y sin red. Devuelve el estado que exigirán las celdas con modelos reales."""
    deps = cfg["dependencies"]
    probs, unc, checked = [], [], []
    for name, spec in deps["required"].items():
        v = version_fn(name)
        if v is None:
            probs.append(f"{name} no está instalado (requerido {spec or 'cualquier versión'}). No se instala automáticamente.")
            continue
        ok, err = _in_spec(v, spec, name)
        if err:
            unc.append(err)
        elif not ok:
            probs.append(f"{name} {v} fuera del rango aceptado {spec}. No se cambia automáticamente.")
        else:
            checked.append(f"{name} {v} ∈ '{spec or '*'}'")
    for dist in deps["verify_declared_requirements_of"]:
        if version_fn(dist) is None:
            continue  # ya informado arriba como no instalado
        p, u, c = check_declared_requirements(dist, version_fn, requires_fn)
        probs, unc, checked = probs + p, unc + u, checked + c

    optional = {}
    for name, spec in deps.get("optional", {}).items():
        v = version_fn(name)
        entry = {"installed": v, "spec": spec, "available": False, "problems": []}
        if v is not None:
            ok, err = _in_spec(v, spec, name)
            p, u, _ = check_declared_requirements(name, version_fn, requires_fn)
            entry["problems"] = ([err] if err else []) + ([] if ok or err else [f"{name} {v} fuera de {spec}"]) + p + u
            entry["available"] = not entry["problems"]
        optional[name] = entry

    versions = installed_versions(version_fn=version_fn)
    return {"ok": not probs and not unc, "problems": probs, "uncertain": unc, "checked": checked,
            "optional": optional, "versions": versions, "fingerprint": fingerprint(versions),
            "config_sha256": cfg["_sha256"], "validated_at": now_iso(), "network_used": False}


def import_smoke(state):
    """Prueba de importación REAL de lo que usarán las celdas 6 y 7 (sin red ni descargas): detecta
    incompatibilidades que los metadatos no declaran (p.ej. torchao). Añade el resultado a `state`."""
    msgs, probs = [], []
    try:
        from .compat import neutralize_incompatible_torchao
        msgs.append(neutralize_incompatible_torchao())
        from diffusers import AutoPipelineForImage2Image, EulerDiscreteScheduler  # noqa: F401
        from transformers import CLIPModel, CLIPProcessor  # noqa: F401
        msgs.append("diffusers (AutoPipelineForImage2Image, EulerDiscreteScheduler) y transformers (CLIP) importan bien.")
    except Exception as e:  # cualquier fallo de importación bloquea las celdas 6 y 7
        probs.append(f"Importación fallida: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
    state["import_smoke"] = {"ok": not probs, "messages": msgs, "problems": probs}
    state["problems"] = state["problems"] + probs
    state["ok"] = state["ok"] and not probs
    return state


def save_validation(state, path="env/validation.json"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


class ValidationError(RuntimeError):
    pass


def require_validation(cfg, path="env/validation.json", version_fn=_version):
    """Lo llaman las celdas 6 y 7 (y experiments.pilot con backend real). Lanza ValidationError si
    la validación no existe, falló, es de otra configuración o el entorno cambió desde entonces."""
    p = Path(path)
    if not p.exists():
        raise ValidationError("No hay validación de compatibilidad: ejecute primero la celda 4.")
    state = json.loads(p.read_text(encoding="utf-8"))
    if not state.get("ok"):
        raise ValidationError("La validación de la celda 4 no fue correcta; revise sus problemas e incertidumbres.")
    if state.get("config_sha256") != cfg["_sha256"]:
        raise ValidationError("La configuración cambió después de validar: vuelva a ejecutar la celda 4.")
    if state.get("fingerprint") != fingerprint(installed_versions(version_fn=version_fn)):
        raise ValidationError("Los paquetes instalados cambiaron después de validar: vuelva a ejecutar la celda 4.")
    return state
