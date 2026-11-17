"""Estilo del cliente: se deduce de lo que acepta y rechaza (y de sus motivos) y retroalimenta la ronda siguiente.

Las etiquetas (color, corte, estilo) son campos opcionales de cada prenda en config/vitrina.yaml.
"""
CAMPOS = ("color", "corte", "estilo")


def etiquetas(cfg, items, campos=CAMPOS):
    """Etiquetas de un look, p. ej. ["color negro", "corte recto", "estilo formal"]. Prendas sin etiqueta: nada."""
    cat = cfg["catalogo"]
    return sorted({f"{c} {cat[i][c]}" for i in items if i in cat for c in campos if cat[i].get(c)})
