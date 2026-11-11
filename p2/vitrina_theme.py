"""Identidad visual de Vitrina IA: «la cinta métrica».

Un probador de tienda de ropa tiene un objeto característico: la cinta de sastre, amarilla y con marcas. Es la idea
única de la página: una cinta recorre la cabecera con las marcas 1 a 5 (los cinco looks) y cada look cuelga bajo su
propio tramo de cinta. El resto es quieto: fondo gris espejo, tinta, un solo acento (el amarillo de la cinta),
esquinas casi rectas y sin sombras. Gradio solo pone la estructura; aquí se define cómo se ve.
"""
import base64
import html
from pathlib import Path

def _font_faces():
    """@font-face con las tipografías del proyecto (p2/assets/fonts, fuentes variables, licencia OFL) incrustadas:
    la interfaz se ve igual sin internet. Sin los archivos, cae a Arial Narrow / system-ui (peor, pero funciona)."""
    d = Path(__file__).parent / "assets" / "fonts"
    out = []
    for nombre, archivo, peso in (("Big Shoulders Display", "BigShouldersDisplay-latin.woff2", "700 900"),
                                  ("Hanken Grotesk", "HankenGrotesk-latin.woff2", "400 700")):
        f = d / archivo
        if f.exists():
            b64 = base64.b64encode(f.read_bytes()).decode("ascii")
            out.append(f"@font-face {{ font-family: '{nombre}'; font-style: normal; font-weight: {peso}; "
                       f"font-display: swap; src: url(data:font/woff2;base64,{b64}) format('woff2'); }}")
    return "\n".join(out) + "\n"


_CSS = """
:root, .dark {
  --piso: #E6E9E6; --papel: #F7F8F6; --tinta: #15191C; --cinta: #EFC21B;
  --acero: #566067; --linea: #BEC5C1; --rechazo: #A8341F;
  --f-display: 'Big Shoulders Display', 'Arial Narrow', 'Roboto Condensed', sans-serif;
  color-scheme: light;
}
body, .gradio-container { background: var(--piso) !important; color: var(--tinta); }
.gradio-container { max-width: 1240px !important; margin: auto; padding: 0 clamp(14px, 4vw, 28px) 56px !important; font-size: 15.5px; }
footer { display: none !important; }
.gradio-container .prose, .gradio-container .md { color: var(--tinta); }

/* ---------- cabecera: la cinta ---------- */
#hero { margin-top: 26px; }
.tape { position: relative; display: flex; height: 52px; padding-left: 16px; background: var(--cinta);
        border: 2px solid var(--tinta); }
.tape-tip { position: absolute; left: -2px; top: -2px; bottom: -2px; width: 16px; background: var(--tinta); }
.tape span { flex: 1; padding: 4px 0 0 9px; font: 800 1.6rem/1 var(--f-display); color: var(--tinta);
  background-image: linear-gradient(90deg, var(--tinta) 0 2px, transparent 2px),
                    repeating-linear-gradient(90deg, var(--tinta) 0 1px, transparent 1px 10px);
  background-size: 100% 22px, 100% 10px; background-repeat: no-repeat; background-position: left bottom; }
.hero-body { display: flex; align-items: flex-end; justify-content: space-between; gap: 12px 40px; flex-wrap: wrap;
             padding: 22px 0 4px; }
#hero h1 { margin: 0; font: 900 clamp(4.4rem, 12vw, 9rem)/.8 var(--f-display); letter-spacing: -.01em;
           color: var(--tinta); }
.hero-copy { max-width: 38ch; font-size: 1.05rem; line-height: 1.5; padding-bottom: 6px; }
.hero-copy p { margin: 0; }
.hero-copy small { display: block; margin-top: 10px; color: var(--acero); font-size: .8rem; }

/* ---------- ejemplo de foto ---------- */
.ejemplo { display: flex; flex-wrap: wrap; gap: 16px 22px; align-items: flex-start; margin-top: 26px; padding: 16px 18px;
           background: var(--papel); border: 2px solid var(--tinta); }
.ejemplo figure { margin: 0; flex: none; width: 104px; }
.ejemplo img { display: block; width: 100%; height: auto; border: 1.5px solid var(--linea); }
.ejemplo figcaption { margin-top: 4px; font: 800 1rem/1 var(--f-display); letter-spacing: .02em; }
.ejemplo h3 { margin: 0 0 6px; font: 800 1.5rem/1 var(--f-display); color: var(--tinta); }
.ejemplo ul { margin: 0; padding-left: 18px; font-size: .95rem; line-height: 1.5; color: var(--tinta); }
.ejemplo .nota { margin-top: 8px; }

/* ---------- títulos de sección (los pasos sí son una secuencia) ---------- */
.sec { display: flex; align-items: center; gap: 12px; margin: 40px 0 16px; }
.sec b { min-width: 36px; padding: 6px 10px 4px; text-align: center; background: var(--cinta);
         border: 2px solid var(--tinta); font: 800 1.2rem/1 var(--f-display); }
.sec h2 { margin: 0; font: 800 2rem/1 var(--f-display); letter-spacing: .005em; color: var(--tinta); }
.nota { margin: 6px 0 0; color: var(--acero); font-size: .88rem; line-height: 1.45; }

/* ---------- campos: sin píldoras, etiquetas discretas ---------- */
.gradio-container .block { box-shadow: none !important; }
/* Gradio rellena los contenedores de formulario con el color de borde (para dibujar las líneas entre campos):
   aquí no hay líneas, solo aire */
.gradio-container .form { background: transparent !important; border: 0 !important; gap: 16px !important; }
.gradio-container .form > .block, .gradio-container .block.padded { background: transparent !important; }
.gradio-container label > span:first-child, .gradio-container .block-title, .gradio-container [data-testid="block-label"] {
  font-weight: 700; color: var(--tinta); }
.gradio-container [data-testid="block-info"] { color: var(--acero); }
.gradio-container input[type=checkbox] { accent-color: var(--tinta); width: 18px; height: 18px; }
.gradio-container textarea, .gradio-container input[type=text] { border: 1.5px solid var(--linea) !important;
  border-radius: 2px !important; background: var(--papel) !important; }
.gradio-container textarea:focus, .gradio-container input[type=text]:focus { border-color: var(--tinta) !important;
  outline: 3px solid var(--cinta); outline-offset: 1px; }

/* consentimiento de la API: solo se ve cuando está elegido el enfoque API (CSS puro, sin evento del servidor) */
#consent-api { display: none !important; }
body:has(#approach input[value*="API de im"]:checked) #consent-api { display: block !important; }

/* ---------- opciones (radios) como piezas rectas ---------- */
.gradio-container [data-testid$="-radio-label"] { position: relative; display: inline-flex; align-items: center;
  justify-content: center; margin: 0 6px 6px 0; padding: 9px 16px; border: 1.5px solid var(--tinta) !important;
  border-radius: 2px !important; background: transparent !important; color: var(--tinta) !important;
  font-weight: 600; cursor: pointer; box-shadow: none !important; transition: background .15s, color .15s; }
.gradio-container [data-testid$="-radio-label"] span { color: inherit !important; }
.gradio-container [data-testid$="-radio-label"] input { position: absolute; width: 1px; height: 1px; margin: 0;
  opacity: 0; pointer-events: none; }  /* oculto pero enfocable: el clic lo recibe la etiqueta */
.gradio-container [data-testid$="-radio-label"]:hover { background: rgba(21, 25, 28, .07) !important; }
.gradio-container [data-testid$="-radio-label"]:has(input:checked) { background: var(--tinta) !important;
  color: var(--cinta) !important; }
.gradio-container [data-testid$="-radio-label"]:has(input[value="rechazada"]:checked) {
  background: var(--rechazo) !important; border-color: var(--rechazo) !important; color: #fff !important; }
.gradio-container [data-testid$="-radio-label"]:has(input:focus-visible) { outline: 3px solid var(--cinta);
  outline-offset: 2px; box-shadow: 0 0 0 5px var(--tinta) !important; }
.decide > [data-testid="block-info"] { position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; }   /* la etiqueta se queda para lectores de pantalla, no se ve */
.decide .wrap { display: flex; gap: 0; flex-wrap: nowrap; }
.decide [data-testid$="-radio-label"] { flex: 1; margin: 0; padding: 11px 6px; }
.decide [data-testid$="-radio-label"]:last-child { margin-left: -1.5px; }

/* ---------- botones ---------- */
.gradio-container button { white-space: nowrap; }
.gradio-container button.primary { background: var(--cinta) !important; color: var(--tinta) !important;
  border: 2px solid var(--tinta) !important; border-radius: 2px !important; box-shadow: none !important;
  font: 800 1.2rem/1 var(--f-display) !important; letter-spacing: .03em; transition: background .15s, color .15s; }
.gradio-container button.primary:hover { background: var(--tinta) !important; color: var(--cinta) !important; }
.gradio-container button.secondary { background: transparent !important; color: var(--tinta) !important;
  border: 1.5px solid var(--tinta) !important; border-radius: 2px !important; box-shadow: none !important;
  font-weight: 700 !important; transition: background .15s; }
.gradio-container button.secondary:hover { background: rgba(21, 25, 28, .08) !important; }
.gradio-container button.stop { background: transparent !important; color: var(--rechazo) !important;
  border: 1.5px solid var(--rechazo) !important; border-radius: 2px !important; box-shadow: none !important; }
.gradio-container button.stop:hover { background: var(--rechazo) !important; color: #fff !important; }
.gradio-container button:focus-visible { outline: 3px solid var(--cinta) !important; outline-offset: 2px;
  box-shadow: 0 0 0 5px var(--tinta) !important; }

/* ---------- los cinco looks: cada uno cuelga de su tramo de cinta ---------- */
.look { margin: 0; animation: entra .38s ease both; animation-delay: calc(var(--i, 0) * 70ms); }
.look-tape { display: flex; justify-content: space-between; align-items: flex-start; height: 42px;
  padding: 5px 10px 0; background-color: var(--cinta); border: 2px solid var(--tinta); border-bottom: 0;
  background-image: repeating-linear-gradient(90deg, var(--tinta) 0 1px, transparent 1px 8px);
  background-size: 100% 9px; background-repeat: no-repeat; background-position: left bottom; }
.look-tape b { font: 900 1.45rem/1 var(--f-display); }
.look-tape span { font-size: .78rem; font-weight: 600; padding-top: 4px; }
.look-tape .est { display: none; font-weight: 700; }
.look-photo { position: relative; aspect-ratio: 2 / 3; overflow: hidden; border: 2px solid var(--tinta);
  background: var(--papel); }
.look-photo img { display: block; width: 100%; height: 100%; object-fit: cover; }
.look figcaption { padding: 10px 2px 0; font-weight: 700; font-size: 1rem; line-height: 1.3 !important;
  height: calc(3 * 1.3em + 10px); box-sizing: border-box; display: -webkit-box !important; -webkit-box-orient: vertical !important;
  -webkit-line-clamp: 3 !important; line-clamp: 3 !important; overflow: hidden !important; }
/* altura fija de 3 líneas (el nombre completo va en el atributo title): las decisiones de los cinco looks quedan
   alineadas y pegadas a su foto */

/* ---------- la decisión se marca en la propia tarjeta (con texto, no solo con color) ---------- */
.look-col:has(input[value="aceptada"]:checked) .look-tape { background-color: var(--tinta); color: var(--cinta);
  background-image: repeating-linear-gradient(90deg, var(--cinta) 0 1px, transparent 1px 8px); }
.look-col:has(input[value="rechazada"]:checked) .look-tape { background-color: var(--rechazo); color: #fff;
  background-image: repeating-linear-gradient(90deg, #fff 0 1px, transparent 1px 8px); }
/* Gradio fuerza un color al texto de dentro de sus bloques HTML: se fija explícitamente */
.look-col:has(input[value="aceptada"]:checked) .look-tape b,
.look-col:has(input[value="aceptada"]:checked) .look-tape span { color: var(--cinta) !important; }
.look-col:has(input[value="rechazada"]:checked) .look-tape b,
.look-col:has(input[value="rechazada"]:checked) .look-tape span { color: #fff !important; }
.look-col:has(input[value="aceptada"]:checked) .look-tape .ronda,
.look-col:has(input[value="rechazada"]:checked) .look-tape .ronda { display: none; }
.look-col:has(input[value="aceptada"]:checked) .look-tape .est-ok,
.look-col:has(input[value="rechazada"]:checked) .look-tape .est-no { display: inline; }
.look-col:has(input[value="rechazada"]:checked) .look-photo img { opacity: .38; filter: grayscale(1); }
.look-col:has(input[value="rechazada"]:checked) .look-photo::after { content: ""; position: absolute; inset: 0;
  background: linear-gradient(to top right, transparent calc(50% - 1.5px), var(--rechazo) calc(50% - 1.5px),
                              var(--rechazo) calc(50% + 1.5px), transparent calc(50% + 1.5px)); }

/* ---------- mientras se generan los looks, la cinta «mide» marca por marca ---------- */
.nota-envio { margin: 10px 0 0; color: var(--acero); font-size: .9rem; line-height: 1.45; }
.generando-msg { display: none; margin: 10px 0 0; font-weight: 700; font-size: .95rem; }
body.generando .generando-msg { display: block; }
@keyframes mide { 0%, 100% { background-color: var(--cinta); color: var(--tinta); }
                  35% { background-color: var(--tinta); color: var(--cinta); } }
@keyframes respira { 50% { border-color: var(--acero); } }
body.generando .tape span { animation: mide 2.6s ease-in-out infinite; }
body.generando .tape span:nth-child(3) { animation-delay: .3s; }
body.generando .tape span:nth-child(4) { animation-delay: .6s; }
body.generando .tape span:nth-child(5) { animation-delay: .9s; }
body.generando .tape span:nth-child(6) { animation-delay: 1.2s; }
body.generando .look.vacio .look-photo { animation: respira 1.6s ease-in-out infinite; }
.look-warn { margin-top: 4px; color: var(--rechazo); font-size: .85rem; font-weight: 600; }
.look.vacio .look-tape { background-color: transparent; border-color: var(--linea); color: var(--acero);
  background-image: repeating-linear-gradient(90deg, var(--linea) 0 1px, transparent 1px 8px); }
.look.vacio .look-photo { display: flex; align-items: center; justify-content: center; background: transparent;
  border: 2px dashed var(--linea); }
.look.vacio .look-photo b { font: 900 7rem/1 var(--f-display); color: var(--linea); }
.look.vacio figcaption { color: var(--acero); font-weight: 500; }
@keyframes entra { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

/* ---------- pestañas ---------- */
.gradio-container [role=tablist] { border-bottom: 2px solid var(--tinta) !important; gap: 2px; }
.gradio-container [role=tab] { margin-bottom: -2px; padding: 10px 16px 6px !important; background: transparent !important;
  border: 0 !important; border-bottom: 6px solid transparent !important; border-radius: 0 !important;
  color: var(--acero) !important; font: 800 1.1rem/1 var(--f-display) !important; letter-spacing: .03em; }
.gradio-container [role=tab][aria-selected=true] { color: var(--tinta) !important; border-bottom-color: var(--cinta) !important; }
.gradio-container [role=tab]:hover { color: var(--tinta) !important; }

#aviso { margin-top: 34px; padding-top: 14px; max-width: 84ch; border-top: 1px solid var(--linea);
         color: var(--acero); font-size: .85rem; }

/* OJO: Gradio antepone un prefijo a los selectores dentro de @media; los que empiezan con .gradio-container no coinciden */
@media (max-width: 720px) {
  button { white-space: normal; }
  /* Gradio da a las columnas un ancho mínimo fijo que impedía apilar: formulario en una columna, looks y botones de a dos */
  #form-row { flex-direction: column; }
  #form-row > .column { min-width: 0 !important; width: 100%; }
  #looks-row > .column { min-width: 0 !important; flex: 1 1 calc(50% - 12px) !important; }
  #acciones > button { min-width: 0 !important; flex: 1 1 calc(50% - 8px) !important; }
  .look figcaption { height: calc(4 * 1.3em + 10px); -webkit-line-clamp: 4 !important; line-clamp: 4 !important; }
  .tabs, .tab-container, [role=tablist] { flex-wrap: wrap !important; max-width: 100% !important; min-width: 0 !important; }
  [role=tab] { padding: 8px 10px 4px !important; font-size: 1rem !important; }
  .tape span { font-size: 1.2rem; padding-left: 5px; }
  .sec { margin-top: 30px; } .sec h2 { font-size: 1.6rem; }
  .look.vacio .look-photo b { font-size: 5rem; }
}
@media (prefers-reduced-motion: reduce) {
  .look, .tape span, .look.vacio .look-photo { animation: none !important; }
  *, *::before { transition: none !important; }
}
"""

CSS = _font_faces() + _CSS

_THEME_VARS = {
    "body_background_fill": "#E6E9E6", "body_text_color": "#15191C", "body_text_color_subdued": "#566067",
    "background_fill_primary": "#E6E9E6", "background_fill_secondary": "#E6E9E6", "block_background_fill": "transparent",
    "block_border_width": "0px", "block_border_color": "#BEC5C1", "block_shadow": "none", "block_padding": "10px 0",
    "block_label_background_fill": "transparent", "block_label_text_color": "#15191C", "block_label_border_width": "0px",
    "block_label_shadow": "none", "block_label_padding": "0 0 6px 0", "block_label_text_weight": "700",
    "block_title_background_fill": "transparent", "block_title_text_color": "#15191C", "block_title_padding": "0 0 6px 0",
    "block_title_text_weight": "700", "block_info_text_color": "#566067",
    "input_background_fill": "#F7F8F6", "input_border_color": "#BEC5C1", "input_border_width": "1.5px",
    "input_shadow": "none", "panel_background_fill": "transparent", "border_color_primary": "#BEC5C1",
    "button_primary_background_fill": "#EFC21B", "button_primary_background_fill_hover": "#15191C",
    "button_primary_text_color": "#15191C", "button_primary_text_color_hover": "#EFC21B",
    "button_primary_border_color": "#15191C", "button_secondary_background_fill": "transparent",
    "button_secondary_text_color": "#15191C", "button_secondary_border_color": "#15191C",
    "checkbox_background_color_selected": "#15191C", "checkbox_border_color_selected": "#15191C",
    "table_even_background_fill": "#F7F8F6", "table_odd_background_fill": "#E6E9E6",
}


def theme():
    """Tema base de Gradio con la paleta de la cinta; el modo oscuro usa los mismos valores (la página es clara)."""
    import gradio as gr
    cinta = gr.themes.Color(c50="#FFF9DB", c100="#FDF0A8", c200="#F8E070", c300="#F4D13D", c400="#EFC21B", c500="#D9AB0C",
                            c600="#B38A08", c700="#8A6A06", c800="#614A05", c900="#3D2F03", c950="#241B02")
    t = gr.themes.Base(primary_hue=cinta, neutral_hue="slate", radius_size=gr.themes.sizes.radius_sm,
                       font=["Hanken Grotesk", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
                       font_mono=["ui-monospace", "Consolas", "monospace"])
    import inspect
    ok = set(inspect.signature(t.set).parameters)  # no todas las variables tienen versión oscura
    dark = {k + "_dark": v for k, v in _THEME_VARS.items() if k + "_dark" in ok}
    return t.set(**{k: v for k, v in _THEME_VARS.items() if k in ok}, **dark)


# ------------------------------------------------------------------ piezas HTML
def hero_html(version):
    marcas = "".join(f"<span>{k}</span>" for k in range(1, 6))
    return ("<header id='hero'><div class='tape' aria-hidden='true'><i class='tape-tip'></i>" + marcas + "</div>"
            "<div class='hero-body'><h1>Vitrina IA</h1><div class='hero-copy'>"
            "<p>Probador virtual de estilos. Sube tu foto y mírate con cinco looks de la tienda. Tú decides cuáles te "
            f"gustan; después una IA te explica tu elección.</p><small>versión {html.escape(str(version))}</small>"
            "</div></div></header>")


def ejemplo_html():
    """Ejemplo de foto buena (p2/assets/ejemplos) con las pautas de pose. Sin el archivo, solo las pautas."""
    d = Path(__file__).parent / "assets" / "ejemplos"
    img = ""
    for archivo, pie, alt in (("pose_ejemplo.jpg", "Ideal", "de pie, de frente, cuerpo completo y brazos separados"),
                              ("pose_ejemplo_2.jpg", "También sirve", "cuerpo completo, con una pose más natural")):
        if (d / archivo).exists():
            b64 = base64.b64encode((d / archivo).read_bytes()).decode("ascii")
            img += (f"<figure><img src='data:image/jpeg;base64,{b64}' alt='Ejemplo de foto: {alt}'/>"
                    f"<figcaption>{pie}</figcaption></figure>")
    return ("<aside class='ejemplo'>" + img + "<div><h3>Cómo tomar tu foto</h3><ul>"
            "<li>De pie y de frente, con el cuerpo completo: de la cabeza a los zapatos.</li>"
            "<li>Lo ideal es tener los brazos un poco separados del cuerpo; las manos a la vista y sin tapar la ropa.</li>"
            "<li>Solo tú, con luz pareja y un fondo liso.</li></ul>"
            "<p class='nota'>Importa sobre todo en la generación local (intermedio), que detecta tu pose para "
            "cambiar solo la ropa.</p></div></aside>")


def sec(n, titulo):
    return f"<div class='sec'><b>{n}</b><h2>{html.escape(titulo)}</h2></div>"


def empty_card(k):
    return (f"<figure class='look vacio'><div class='look-tape'><b>V{k}</b></div>"
            f"<div class='look-photo'><b>{k}</b></div><figcaption>Tu look {k} aparecerá aquí.</figcaption></figure>")


def look_card(i, vid, nombre, ronda, img_path, aviso=""):
    """Un look colgado de su tramo de cinta. i = posición (0-4), solo para escalonar la entrada."""
    b64 = base64.b64encode(Path(img_path).read_bytes()).decode("ascii")
    warn = f"<div class='look-warn'>{html.escape(aviso)}</div>" if aviso else ""
    return (f"<figure class='look' style='--i:{i}'><div class='look-tape'><b>{html.escape(vid)}</b>"
            f"<span class='ronda'>ronda {ronda}</span><span class='est est-ok'>Aceptado</span>"
            f"<span class='est est-no'>Descartado</span></div><div class='look-photo'>"
            f"<img src='data:image/png;base64,{b64}' alt='{html.escape(vid)}: {html.escape(nombre)}'/></div>"
            f"<figcaption title='{html.escape(nombre)}'>{html.escape(nombre)}{warn}</figcaption></figure>")
