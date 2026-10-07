"""The infrastructure state map through the bindings."""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys

import pytest

s = importlib.import_module("_pypto_serving")


def _load_infra():
    """``pypto_serving.cpp.infra`` needs no torch, but importing it by name pulls
    ``pypto_serving/__init__.py``, which does. Load it by path instead."""
    path = pathlib.Path(__file__).resolve().parents[3] / "pypto_serving" / "cpp" / "infra.py"
    spec = importlib.util.spec_from_file_location("cpp_infra", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cpp_infra"] = module
    spec.loader.exec_module(module)
    return module


infra = _load_infra()


@pytest.fixture
def dev(tmp_path) -> str:
    """A directory shaped like /dev on a node with eight cards."""
    for i in range(8):
        (tmp_path / f"davinci{i}").touch()
    for control in ("davinci_manager", "devmm_svm", "hisi_hdc"):
        (tmp_path / control).touch()
    return str(tmp_path)


def test_discovery_finds_cards_and_skips_control_nodes(dev):
    assert s.discover_devices(dev) == list(range(8))
    assert s.discover_devices(str(pathlib.Path(dev) / "nope")) == []


def test_a_node_describes_what_it_owns(dev):
    node = s.local_node("hg-atlas-01", "192.168.150.11", dev)
    assert node.name == "hg-atlas-01"
    assert node.owns(4) and not node.owns(8)


def test_a_replica_takes_a_card_and_releases_it_when_it_dies(dev):
    m = s.StateMap()
    m.add_node(s.local_node("hg-atlas-01", "192.168.150.11", dev))
    m.add_replica(s.Replica("qwen-0", node="hg-atlas-01", devices=[4], port=8000, phase=s.Phase.ACTIVE))

    assert s.validate(m) == []
    assert m.devices_free("hg-atlas-01") == [0, 1, 2, 3, 5, 6, 7]

    # Draining still serves what it has, so the card is still taken.
    m.set_phase("qwen-0", s.Phase.DRAINING)
    assert m.devices_in_use("hg-atlas-01") == [4]

    # Dead has released it, which is what lets capacity be rebuilt there.
    m.set_phase("qwen-0", s.Phase.DEAD)
    assert m.devices_free("hg-atlas-01") == list(range(8))


def test_the_version_moves_on_every_edit(dev):
    m = s.StateMap()
    assert m.version == 0
    m.add_node(s.local_node("n", "h", dev))
    first = m.version
    m.add_replica(s.Replica("r", node="n", devices=[0], phase=s.Phase.ACTIVE))
    assert m.version > first


def test_validate_catches_a_card_the_node_does_not_have():
    m = s.StateMap()
    m.add_node(s.Node("small", "10.0.0.1", [0, 1]))
    m.add_replica(s.Replica("r", node="small", devices=[7], phase=s.Phase.ACTIVE))
    assert any("device 7 is not on node 'small'" in p for p in s.validate(m))


def test_validate_catches_two_replicas_on_one_card(dev):
    m = s.StateMap()
    m.add_node(s.local_node("n", "h", dev))
    m.add_replica(s.Replica("a", node="n", devices=[4], port=8000, phase=s.Phase.ACTIVE))
    m.add_replica(s.Replica("b", node="n", devices=[4], port=8001, phase=s.Phase.ACTIVE))
    assert any("claimed by a, b" in p for p in s.validate(m))


def test_reconcile_reports_the_gap_not_the_how(dev):
    m = s.StateMap()
    m.add_node(s.local_node("n", "h", dev))
    m.add_replica(s.Replica("r", node="n", devices=[0], phase=s.Phase.ACTIVE))
    assert s.reconcile(m) == []

    m.set_phase("r", s.Phase.DEAD)
    actions = s.reconcile(m)
    assert len(actions) == 1 and actions[0].replica == "r" and actions[0].action == s.Action.START


def test_granted_device_reads_the_brokers_physical_card(monkeypatch):
    monkeypatch.setenv(infra.PHYSICAL_DEVICE_ENV, "4")
    assert infra.granted_device() == 4
    # Off the broker there is nothing to read, and the caller's own value stands.
    monkeypatch.delenv(infra.PHYSICAL_DEVICE_ENV, raising=False)
    assert infra.granted_device(default=7) == 7
    monkeypatch.setenv(infra.PHYSICAL_DEVICE_ENV, "not-a-number")
    assert infra.granted_device(default=7) == 7


def test_describe_replica_uses_the_granted_card_not_the_logical_one(monkeypatch):
    # The process sees logical 0; the map must record the card the node owns.
    monkeypatch.setenv(infra.PHYSICAL_DEVICE_ENV, "4")
    r = infra.describe_replica("qwen-0", node="hg-atlas-01", port=8000, device=0)
    assert r.devices == [4]
    assert r.node == "hg-atlas-01" and r.port == 8000


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------


def _deployment(dev: str):
    m = s.StateMap()
    m.add_node(s.local_node("hg-atlas-01", "192.168.150.11", dev))
    m.add_replica(s.Replica("qwen-0", node="hg-atlas-01", devices=[4], port=8000, phase=s.Phase.ACTIVE))
    return m


def test_a_document_round_trips_and_keeps_its_version(dev):
    m = _deployment(dev)
    back = s.read_document(s.write_document(m))

    assert back.version == m.version
    assert [r.name for r in back.replicas] == ["qwen-0"]
    assert back.replica("qwen-0").devices == [4]
    assert back.devices_free("hg-atlas-01") == [0, 1, 2, 3, 5, 6, 7]


def test_the_document_is_strict(dev):
    with pytest.raises(ValueError, match="unknown key"):
        s.read_document('{"replicaz": []}')
    with pytest.raises(ValueError, match="unknown phase"):
        s.read_document('{"replicas": [{"name": "r", "phase": "zombie"}]}')
    with pytest.raises(ValueError):
        s.read_document("not json")


def test_two_processes_agree_through_a_file(dev, tmp_path):
    # The writer: a replica recording where it landed.
    path = str(tmp_path / "deployment.json")
    s.save_document(_deployment(dev), path)

    # The reader: anything that needs to know what runs where.
    loaded = s.load_document(path)
    assert s.validate(loaded) == []
    assert loaded.devices_in_use("hg-atlas-01") == [4]

    # And a phase written by one side is seen by the other.
    loaded.set_phase("qwen-0", s.Phase.DEAD)
    s.save_document(loaded, path)
    assert s.load_document(path).devices_free("hg-atlas-01") == list(range(8))


def test_saving_leaves_no_temporary_behind(dev, tmp_path):
    # Its own directory: the `dev` fixture puts fake device nodes in tmp_path.
    target = tmp_path / "state"
    target.mkdir()
    s.save_document(_deployment(dev), str(target / "deployment.json"))

    # A reader polling the document must never catch it half-written, which is
    # why it is renamed into place; the temporary must not survive either.
    assert sorted(p.name for p in target.iterdir()) == ["deployment.json"]
