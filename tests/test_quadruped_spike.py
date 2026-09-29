import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_go1_foot_contact_spike_passes(tmp_path, monkeypatch):
    import spike_quadruped

    monkeypatch.setattr(spike_quadruped, "OUT", tmp_path)
    s = spike_quadruped.main(video=False)
    assert s["verdict"] == "PASS", s
    assert s["fell_at"] is None
    assert s["fr_foot_block_contact_ticks"] > 5
    assert s["non_foot_block_contacts"] == []
    assert s["block_displacement_m"] > 0.03
