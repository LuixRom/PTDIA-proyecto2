import json
import threading

import pytest

from conftest import image_bytes
from p2 import sus
from p2.decisions import DecisionStore
from p2.generate import mock_generate
from p2.schemas import ExplainResult, Summary
from p2.tempdb import TempDB
from p2.ui import clean_alias, new_session, on_submit, on_summarize, on_sus


@pytest.mark.parametrize("answers,expected", [
    ([3] * 10, 50.0),
    ([5, 1] * 5, 100.0),
    ([1, 5] * 5, 0.0),
    ([4, 2, 4, 2, 4, 2, 4, 2, 4, 2], 75.0),
])
def test_sus_score_known_values(answers, expected):
    assert sus.score(answers) == expected and len(sus.ITEMS) == 10


@pytest.mark.parametrize("answers,msg", [([3] * 9, "10 ítems"), ([3] * 9 + [None], "pregunta 10"),
                                         ([3] * 9 + [6], "entre 1 y 5")])
def test_sus_rejects_invalid(answers, msg):
    with pytest.raises(ValueError, match=msg):
        sus.score(answers)


def _res():
    return ExplainResult(summary=Summary(aceptadas=["V1"], rechazadas=["V2", "V3", "V4", "V5"], resumen="V1 fue aceptada."),
                         verified=True, attempts=2, fallback=False, llm="qwen", raw_responses=["{mal", "{bien}"])


def test_tempdb_roundtrip_and_concurrency(tmp_path):
    db = TempDB(tmp_path / "x" / "agente_tmp.sqlite")
    db.add_summary("s-1", "ana", _res())
    threads = [threading.Thread(target=db.add_sus, args=(f"p{i}", "s-1", [3] * 10, 50.0)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    r = db.rows("resumenes")[0]
    assert (r["session_id"], r["participante"], r["verificado"], r["respaldo"], r["intentos"]) == ("s-1", "ana", 1, 0, 2)
    assert json.loads(r["respuestas_llm"]) == ["{mal", "{bien}"] and json.loads(r["rechazadas"]) == ["V2", "V3", "V4", "V5"]
    assert len(db.rows("sus")) == 20
    with pytest.raises(ValueError):
        db.rows("sqlite_master; DROP TABLE sus")


def test_alias_is_cleaned_and_bounded():
    assert clean_alias("  Ana   B \n") == "Ana B" and len(clean_alias("x" * 100)) == 40 and clean_alias(None) == ""


def test_app_flow_writes_db_and_sus(tmp_path, cfg):
    store = DecisionStore(tmp_path / "sesiones")
    db = TempDB(store.root / "agente_tmp.sqlite")
    sid, _ = new_session(store, image_bytes(), cfg, "A2", mock_generate, store.root, participant="  luis  ")
    on_submit(store, sid, ["aceptada", "rechazada", "aceptada", "rechazada", "aceptada"], [""] * 5)
    res = on_summarize(store, sid, db=db)
    assert store.load(sid).participant == "luis" and db.rows("resumenes")[0]["participante"] == "luis" and res.verified
    assert on_sus(store, sid, "luis", [5, 1] * 5, db) == 100.0
    line = json.loads((store.root / "sus.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["score"] == 100.0 and line["participant"] == "luis" and db.rows("sus")[0]["puntuacion"] == 100.0
    with pytest.raises(ValueError):
        on_sus(store, sid, "luis", [5] * 9 + [None], db)
    assert len(db.rows("sus")) == 1  # un SUS incompleto no se guarda
