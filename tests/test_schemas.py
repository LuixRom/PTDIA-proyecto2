import pytest
from pydantic import ValidationError

from conftest import SHA, make_session
from p2.schemas import RunRecord, Session, Variation


def _v(i, **kw):
    return Variation(**{"id": f"V{i}", "method": "A2", "attribute": "a", "prompt": "p", "seed": i, **kw})


def test_valid_session_roundtrip():
    s = make_session({"V1": "aceptada"})
    assert Session.model_validate_json(s.model_dump_json()) == s


@pytest.mark.parametrize("n", [4, 6])
def test_rejects_not_five(n):
    with pytest.raises(ValidationError, match="exactamente 5"):
        Session(session_id="x", input_sha256=SHA, variations=[_v((i % 5) + 1) for i in range(n)])


def test_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="sin repetir"):
        Session(session_id="x", input_sha256=SHA, variations=[_v(1), _v(1), _v(3), _v(4), _v(5)])


@pytest.mark.parametrize("bad", [{"decision": "quizas"}, {"prompt": ""}, {"id": "V6"}, {"seed": None}])
def test_rejects_bad_variation(bad):
    with pytest.raises(ValidationError):
        _v(1, **bad)


def test_rejects_bad_sha():
    with pytest.raises(ValidationError):
        Session(session_id="x", input_sha256="123", variations=[_v(i) for i in range(1, 6)])


def test_runrecord_rejects_bad_strength():
    base = dict(run_id="r", method="A1", variation="V1", attribute="a", prompt="p", negative_prompt="n", seed=1,
                steps=20, guidance_scale=7.0, scheduler="euler", width=512, height=512, model="m",
                input_sha256=SHA, output="o.png", output_sha256=SHA, time_s=1.0)
    RunRecord(strength=0.4, **base)
    with pytest.raises(ValidationError):
        RunRecord(strength=1.5, **base)
