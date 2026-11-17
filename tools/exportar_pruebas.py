"""Evidencia de las pruebas con participantes: lee todas las sesiones de una carpeta y genera un Excel (y un JSON
con los mismos datos). Sin imágenes.

Uso:
    .venv\\Scripts\\python.exe tools\\exportar_pruebas.py C:\\PTDIA\\sesiones_local            -> pruebas.xlsx y pruebas.json ahí
    .venv\\Scripts\\python.exe tools\\exportar_pruebas.py CARPETA --salida informe\\pruebas.xlsx

Hojas: resumen (medias entre participantes, ronda 1 frente a ronda 2), sesiones, decisiones, perfiles y SUS.
Contiene nombres de participantes: no lo subas al repositorio.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def _media(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def recolectar(carpeta, cfg):
    """{hoja: [filas]} a partir de las sesiones de `carpeta` (subcarpetas con decisions.json)."""
    from p2 import estilo, vitrina
    from p2.decisions import DecisionStore
    store = DecisionStore(carpeta)
    sus = _jsonl(Path(carpeta) / "sus.jsonl")
    sus_por_sesion = {x["session_id"]: x["score"] for x in sus}
    hojas = {"sesiones": [], "decisiones": [], "perfiles": [], "SUS": []}
    for d in sorted(p for p in Path(carpeta).iterdir() if (p / "decisions.json").exists()):
        sid = d.name
        s = store.load(sid)
        m = vitrina.session_metrics(store, sid, cfg)
        o = s.options
        tasas = m["tasa_aceptacion_por_ronda"]
        hojas["sesiones"].append({
            "sesion": sid, "participante": s.participant, "creada": s.created_at, "linea": o.get("linea"),
            "ocasion": o.get("occasion"), "modo": o.get("mode"), "enfoque": o.get("approach"), "rondas": m["rondas"],
            "rondas_con_estilo": m["rondas_con_estilo"], **{f"aceptacion_{k}": v for k, v in tasas.items()},
            "respeta_no_quiero_pct": m["ronda_estilo_respeta_no_quiero_pct"],
            "coincide_quiero_pct": m["ronda_estilo_coincide_quiero_pct"],
            "regeneradas_que_estaban_rechazadas_pct": m["regeneradas_que_estaban_rechazadas_pct"],
            "decisiones_registradas": m["decisiones_registradas"], "resumenes_verificados_llm": m["resumenes_verificados_llm"],
            "identidad_ssim_rostro_media": m["identidad_ssim_rostro_media"], "imagenes_generadas": m["imagenes_generadas"],
            "costo_api_estimado_usd": m["costo_api_estimado_usd"], "sus": sus_por_sesion.get(sid)})
        for x in sorted(estilo.decididos(s), key=lambda x: (x["ronda"], x["id"])):
            hojas["decisiones"].append({
                "sesion": sid, "participante": s.participant, "ronda": x["ronda"], "id": x["id"], "look": x["look"],
                "etiquetas": ", ".join(estilo.etiquetas(cfg, x["items"])), "decision": x["decision"], "motivo": x["motivo"]})
        for e in _jsonl(d / "estilos.jsonl"):
            hojas["perfiles"].append({
                "sesion": sid, "participante": s.participant, "ronda": e["round"], "tipo": "deducido",
                "fuente": e["fuente"], "quiero": ", ".join(p["valor"] for p in e["quiero"]),
                "no_quiero": ", ".join(p["valor"] for p in e["no_quiero"]), "explicacion": e["explicacion"]})
        for h in vitrina.history(store, sid):
            if h["kind"] == "ronda_estilo":
                hojas["perfiles"].append({
                    "sesion": sid, "participante": s.participant, "ronda": h["round"], "tipo": "usado en la ronda",
                    "fuente": "editado por el cliente", "quiero": ", ".join(h["estilo"]["quiero"]),
                    "no_quiero": ", ".join(h["estilo"]["no_quiero"]),
                    "explicacion": "; ".join(f"{k}: {v}" for k, v in h.get("por_que", {}).items())})
    for x in sus:
        hojas["SUS"].append({"participante": x["participant"], "sesion": x["session_id"], "fecha": x["at"],
                             **{f"p{i}": a for i, a in enumerate(x["answers"], start=1)}, "puntaje": x["score"]})
    ses = hojas["sesiones"]
    hojas["resumen"] = [
        {"indicador": "participantes (nombres distintos)", "valor": len({x["participante"] for x in ses if x["participante"]})},
        {"indicador": "sesiones", "valor": len(ses)},
        {"indicador": "aceptación media ronda 1", "valor": _media([x.get("aceptacion_r1") for x in ses])},
        {"indicador": "aceptación media ronda 2", "valor": _media([x.get("aceptacion_r2") for x in ses])},
        {"indicador": "sesiones con ronda 2", "valor": sum(x.get("aceptacion_r2") is not None for x in ses)},
        {"indicador": "aceptación ronda 1 (solo sesiones con ronda 2)",
         "valor": _media([x.get("aceptacion_r1") for x in ses if x.get("aceptacion_r2") is not None])},
        {"indicador": "% looks de rondas con estilo que respetan «No quiero» (media)",
         "valor": _media([x["respeta_no_quiero_pct"] for x in ses])},
        {"indicador": "% looks de rondas con estilo con algún «Quiero» (media)",
         "valor": _media([x["coincide_quiero_pct"] for x in ses])},
        {"indicador": "SUS medio", "valor": _media([x["puntaje"] for x in hojas["SUS"]])},
        {"indicador": "nota", "valor": "Exploratorio: muestra pequeña, sin inferencia estadística."},
    ]
    return {k: hojas[k] for k in ("resumen", "sesiones", "decisiones", "perfiles", "SUS")}


def escribir_excel(hojas, salida):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    wb.remove(wb.active)
    for nombre, filas in hojas.items():
        ws = wb.create_sheet(nombre)
        cols = list(dict.fromkeys(k for f in filas for k in f)) or ["(sin datos)"]
        ws.append(cols)
        for c in ws[1]:
            c.font = Font(bold=True)
        for f in filas:
            ws.append([f.get(k) for k in cols])
        for i, k in enumerate(cols, start=1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(60, max(10, len(str(k)) + 2))
    wb.save(salida)
    return Path(salida)


def main(argv=None):
    from p2 import config
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("carpeta", help="carpeta de sesiones (p. ej. C:\\PTDIA\\sesiones_local)")
    ap.add_argument("--salida", help="ruta del .xlsx (por defecto CARPETA/pruebas.xlsx)")
    a = ap.parse_args(argv)
    carpeta = Path(a.carpeta)
    hojas = recolectar(carpeta, config.load(str(ROOT / "config" / "vitrina.yaml")))
    xlsx = escribir_excel(hojas, a.salida or carpeta / "pruebas.xlsx")
    xlsx.with_suffix(".json").write_text(json.dumps(hojas, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{xlsx} y {xlsx.with_suffix('.json').name}: {len(hojas['sesiones'])} sesiones, "
          f"{len(hojas['decisiones'])} decisiones, {len(hojas['SUS'])} cuestionarios SUS")
    return xlsx


if __name__ == "__main__":
    main()
