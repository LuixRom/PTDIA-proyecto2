"""LoRA «Estilo de la tienda» (nivel avanzado) aplicado al UNet SIN la librería peft.

El entrenamiento (tools/build_lora_notebook.py, script oficial de diffusers) guarda, por cada capa de atención,
dos matrices A (r x entrada) y B (salida x r). Aplicarlo es sumar W += escala * (B @ A) (con lora_alpha = rank la
escala es 1). Se guardan los pesos originales para quitarlo: así se compara con y sin LoRA en el mismo pipeline.
"""
import torch


class LoraSwitch:
    def __init__(self, unet, path, scale=1.0, prefix="unet."):
        from safetensors.torch import load_file
        sd = load_file(str(path))
        params = dict(unet.named_parameters())
        self.pairs = {}
        for k, a in sd.items():
            if not k.endswith(".lora_A.weight"):
                continue
            base = k[len(prefix):-len(".lora_A.weight")] + ".weight" if k.startswith(prefix) else None
            if base is None or base not in params:
                raise KeyError(f"El LoRA no coincide con el UNet: {k}")
            self.pairs[base] = (a, sd[k.replace(".lora_A.", ".lora_B.")])
        if not self.pairs:
            raise ValueError("El archivo no contiene pesos LoRA (lora_A / lora_B).")
        self.params, self.scale, self.orig, self.on = params, float(scale), {}, False

    @torch.no_grad()
    def set(self, on, scale=None):
        """Activa (on=True) o quita el LoRA; con scale distinto se vuelve a aplicar con esa escala."""
        if self.on:  # primero vuelve a los pesos originales
            for name, w in self.orig.items():
                self.params[name].data.copy_(w)
            self.on = False
        if scale is not None:
            self.scale = float(scale)
        if not on:
            return self
        for name, (a, b) in self.pairs.items():
            p = self.params[name]
            self.orig.setdefault(name, p.data.detach().to("cpu", copy=True))
            delta = (b.float() @ a.float()) * self.scale
            p.data.add_(delta.to(device=p.device, dtype=p.dtype))
        self.on = True
        return self
