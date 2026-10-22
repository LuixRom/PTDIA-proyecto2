import pytest

from conftest import make_session
from p2.decisions import DecisionStore


@pytest.fixture
def store(tmp_path):
    st = DecisionStore(tmp_path)
    st.create(make_session())
    return st


def test_accept_and_reject_each(store):
    for k in range(1, 6):
        store.decide("s-test", f"V{k}", "aceptada" if k % 2 else "rechazada", f"motivo {k}")
    s = store.load("s-test")
    assert s.accepted() == ["V1", "V3", "V5"] and s.rejected() == ["V2", "V4"]
    assert s.by_id("V2").reason == "motivo 2" and s.by_id("V2").decided_at


def test_summary_ready_only_with_five(store):
    for k in range(1, 5):
        store.decide("s-test", f"V{k}", "aceptada")
    assert not store.summary_ready("s-test")
    store.decide("s-test", "V5", "rechazada")
    assert store.summary_ready("s-test")


def test_change_of_mind_is_traced(store):
    store.decide("s-test", "V1", "aceptada")
    store.decide("s-test", "V1", "rechazada", "cambié de opinión")
    ev = [e for e in store.events("s-test") if e["type"] == "decision"]
    assert [(e["decision"], e["previous"]) for e in ev] == [("aceptada", None), ("rechazada", "aceptada")]
    assert store.load("s-test").by_id("V1").decision == "rechazada"


@pytest.mark.parametrize("vid,dec,err", [("V6", "aceptada", KeyError), ("V1", "talvez", ValueError)])
def test_invalid_inputs(store, vid, dec, err):
    with pytest.raises(err):
        store.decide("s-test", vid, dec)


def test_never_overwrites_existing_session(store):
    with pytest.raises(FileExistsError):
        store.create(make_session())


def test_corrupt_file_is_not_overwritten(store, tmp_path):
    p = tmp_path / "s-test" / "decisions.json"
    p.write_text("{corrupto", encoding="utf-8")
    with pytest.raises(Exception):
        store.decide("s-test", "V1", "aceptada")
    assert p.read_text(encoding="utf-8") == "{corrupto"


def test_atomic_write_leaves_no_tmp(store, tmp_path):
    store.decide("s-test", "V1", "aceptada")
    assert not list((tmp_path / "s-test").glob("*.tmp"))


def test_concurrent_decisions_do_not_lose_updates(store):
    """Gradio atiende cada clic en un hilo: muchas decisiones simultáneas no deben pisarse ni fallar."""
    import threading
    errors = []

    def worker(i):
        try:
            store.decide("s-test", f"V{i % 5 + 1}", "aceptada" if i % 2 else "rechazada", f"m{i}")
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len([e for e in store.events("s-test") if e["type"] == "decision"]) == 40
    assert store.load("s-test").complete()


def test_reason_none_keeps_existing_reason(store):
    store.decide("s-test", "V4", "rechazada", "cambio de género")
    store.decide("s-test", "V4", "rechazada")  # clic tardío sin motivo
    assert store.load("s-test").by_id("V4").reason == "cambio de género"
