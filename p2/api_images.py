"""Enfoque «API» (nivel básico): edición de imágenes con la API de OpenAI (de pago).

La clave se lee de la variable de entorno OPENAI_API_KEY o del archivo .env (nombres aceptados: OPENAI_API_KEY,
API-KEY, API_KEY) y NUNCA se imprime, se registra ni se guarda: todo mensaje de error pasa por ocultar().
Sin dependencias externas (urllib). El .env está en .gitignore.
"""
import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://api.openai.com/v1/images/edits"
KEY_NAMES = ("OPENAI_API_KEY", "API-KEY", "API_KEY")


class ApiError(RuntimeError):
    """Fallo de la API. fatal=True: no tiene sentido seguir (sin saldo, clave inválida)."""

    def __init__(self, msg, fatal=False):
        super().__init__(msg)
        self.fatal = fatal


def cargar_clave(env_path=ROOT / ".env"):
    """Clave desde el entorno o desde .env (sin imprimirla). None si no hay."""
    for n in KEY_NAMES:
        if os.environ.get(n):
            return os.environ[n].strip()
    p = Path(env_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() in KEY_NAMES and v.strip():
                    return v.strip().strip('"').strip("'")
    return None


def ocultar(texto):
    """Quita de un mensaje cualquier cosa con forma de clave (por si la API devuelve un fragmento)."""
    return re.sub(r"sk-[A-Za-z0-9_\-*.]+", "sk-…", str(texto))


def multipart(campos, archivos):
    """(cuerpo, content-type) multipart/form-data sin dependencias externas."""
    b = uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in campos.items():
        out.write(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, (nombre, datos, tipo) in archivos.items():
        out.write(f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{nombre}"\r\n'
                  f'Content-Type: {tipo}\r\n\r\n'.encode() + datos + b"\r\n")
    out.write(f"--{b}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={b}"


def editar(clave, modelo, png, prompt, size, timeout=240):
    """Una llamada a /v1/images/edits. Devuelve el JSON de la respuesta; lanza ApiError."""
    cuerpo, ctype = multipart({"model": modelo, "prompt": prompt, "size": size, "n": "1"},
                              {"image": ("foto.png", png, "image/png")})
    req = urllib.request.Request(ENDPOINT, data=cuerpo, method="POST",
                                 headers={"Authorization": f"Bearer {clave}", "Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:  # noqa: BLE001
            msg = ""
        fatal = e.code in (401, 403) or "credits" in msg.lower() or "billing" in msg.lower()
        raise ApiError(f"HTTP {e.code}: {ocultar(msg)}", fatal=fatal) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ApiError(f"Sin conexión con la API: {ocultar(e)}") from None


def prompt_look(texto):
    return (f"Edit this photo of a person: replace ONLY the clothing so that they wear {texto}. "
            "Keep the person's face, hair, skin tone, hands, shoes, pose, body proportions and the background exactly the same. "
            "If they wear a jacket or blazer, put a plain top underneath and keep the neckline closed and modest. "
            "Photorealistic, natural lighting. Do not add text or logos, and no accessories other than those mentioned.")


class ApiEditor:
    """engine.api(img, texto_del_look) -> (imagen PIL del tamaño recibido, uso informado por la API)."""

    def __init__(self, model="gpt-image-2", size="1024x1536", clave=None):
        self.model, self.size, self.clave = model, size, clave or cargar_clave()

    def available(self):
        return bool(self.clave)

    def __call__(self, img, texto):
        from PIL import Image
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        r = editar(self.clave, self.model, buf.getvalue(), prompt_look(texto), self.size)
        try:
            d = r["data"][0]
            datos = base64.b64decode(d["b64_json"]) if d.get("b64_json") else urllib.request.urlopen(d["url"]).read()
            res = Image.open(io.BytesIO(datos)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            raise ApiError(f"Respuesta de la API no válida: {ocultar(e)}") from None
        return res.resize(img.size, Image.Resampling.LANCZOS), r.get("usage") or {}
