import pytest

from conftest import make_session
from p2.verifier import verificar

GOOD = {"aceptadas": ["V1", "V4"], "rechazadas": ["V2", "V3", "V5"],
        "resumen": "Se aceptaron V1 (luz cálida) y V4 (sonrisa). Se rechazaron V2, V3 y V5."}


def test_correct_summary_passes(mixed):
    assert verificar(GOOD, mixed) == []


def test_state_after_id_passes(mixed):
    s = dict(GOOD, resumen="V1 fue aceptada. V4 fue aceptada. V2 fue rechazada; V3 fue rechazada; V5 fue rechazada.")
    assert verificar(s, mixed) == []


def test_negation_is_understood(mixed):
    s = dict(GOOD, resumen="V1 y V4 fueron aceptadas. V2, V3 y V5 no fueron aceptadas.")
    assert verificar(s, mixed) == []


@pytest.mark.parametrize("change,needle", [
    ({"aceptadas": ["V1", "V2"], "rechazadas": ["V3", "V4", "V5"]}, "'aceptadas'"),       # ID intercambiado
    ({"aceptadas": ["V1"]}, "'aceptadas'"),                                             # ID omitido en la lista
    ({"aceptadas": ["V1", "V4"], "rechazadas": ["V1", "V2", "V3", "V5"]}, "ambas listas"),  # solapados
])
def test_list_errors(mixed, change, needle):
    assert any(needle in e for e in verificar(dict(GOOD, **change), mixed))


def test_invented_id(mixed):
    s = dict(GOOD, resumen=GOOD["resumen"] + " También se aceptó V6.")
    assert any("inexistentes" in e for e in verificar(s, mixed))


def test_contrary_state_in_text(mixed):
    s = dict(GOOD, resumen="Se aceptaron V1, V4 y V2. Se rechazaron V3 y V5.")
    errs = verificar(s, mixed)
    assert any(e.startswith("V2:") and "'aceptada'" in e for e in errs)


def test_missing_mention(mixed):
    s = dict(GOOD, resumen="Se aceptaron V1 y V4. Se rechazaron V2 y V3.")
    assert "V5: no se menciona en el resumen." in verificar(s, mixed)


def test_id_without_state(mixed):
    s = dict(GOOD, resumen="Se aceptaron V1 y V4. Se rechazaron V2, V3. Luego está V5.")
    assert any(e.startswith("V5:") and "explícitamente" in e for e in verificar(s, mixed))


def test_bad_format(mixed):
    assert verificar({"resumen": "x"}, mixed)[0].startswith("Formato inválido")


@pytest.mark.parametrize("state", ["aceptada", "rechazada"])
def test_all_same_edge_cases(state):
    s = make_session({f"V{k}": state for k in range(1, 6)})
    ids = [f"V{k}" for k in range(1, 6)]
    verb = "aceptaron" if state == "aceptada" else "rechazaron"
    summ = {"aceptadas": ids if state == "aceptada" else [], "rechazadas": ids if state == "rechazada" else [],
            "resumen": f"Se {verb} V1, V2, V3, V4 y V5."}
    assert verificar(summ, s) == []


def test_incomplete_session_flagged():
    s = make_session({"V1": "aceptada"})
    assert any("sin decidir" in e for e in verificar(GOOD, s))
