import zipfile

from p2.export import MANIFEST, verify_manifest, write_manifest, zip_dir


def _tree(root):
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.txt").write_text("hola", encoding="utf-8")
    (root / "y.bin").write_bytes(b"\x00\x01")


def test_manifest_roundtrip_and_tamper(tmp_path):
    _tree(tmp_path)
    assert write_manifest(tmp_path) == 2
    assert verify_manifest(tmp_path) == []
    (tmp_path / "y.bin").write_bytes(b"cambiado")
    (tmp_path / "a" / "x.txt").unlink()
    probs = verify_manifest(tmp_path)
    assert "SHA-256 distinto: y.bin" in probs and "Falta a/x.txt" in probs


def test_missing_manifest(tmp_path):
    assert verify_manifest(tmp_path) == [f"Falta {MANIFEST}"]


def test_zip_contains_all_and_is_reproducible(tmp_path):
    src = tmp_path / "src"
    _tree(src)
    write_manifest(src)
    z1 = zip_dir(src, tmp_path / "o1.zip")
    z2 = zip_dir(src, tmp_path / "o2.zip")
    assert sorted(zipfile.ZipFile(z1).namelist()) == ["MANIFEST.sha256", "a/x.txt", "y.bin"]
    assert z1.read_bytes() == z2.read_bytes()
