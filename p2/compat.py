"""Ajustes de compatibilidad en memoria (no instalan, no actualizan ni modifican paquetes).

Caso conocido (Colab, 2026-10): diffusers 0.41.0 hace `from torchao.quantization import FqnToConfig` al
importar sus pipelines si torchao está instalado (quantizers/torchao/torchao_quantizer.py). El torchao
preinstalado en Colab no tiene FqnToConfig -> ImportError al cargar AutoPipelineForImage2Image.
El proyecto no usa cuantización torchao, así que se desactiva SOLO esa integración opcional de diffusers.
"""


def neutralize_incompatible_torchao():
    """Debe llamarse antes de importar pipelines de diffusers. Devuelve un mensaje con lo que hizo."""
    import diffusers.utils.import_utils as iu
    if not iu.is_torchao_available():
        return "torchao no instalado: nada que ajustar."
    try:
        from torchao.quantization import FqnToConfig  # noqa: F401
        return f"torchao {iu._torchao_version} compatible con diffusers: sin cambios."
    except ImportError:
        iu._torchao_available = False  # diffusers consulta is_torchao_available() en cada uso
        return (f"torchao {iu._torchao_version} no tiene FqnToConfig (lo exige diffusers): se desactiva la "
                "integración opcional de cuantización torchao de diffusers, que este proyecto no usa.")
