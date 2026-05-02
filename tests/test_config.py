"""CFG round-trips through YAML and supports immutable updates."""
import os
import tempfile

from av_traits.config import CFG


def test_default_cfg_modalities():
    c = CFG()
    assert c.active_modalities == ("V", "A", "T")
    assert c.num_traits == 5
    assert len(c.trait_names) == c.num_traits


def test_yaml_roundtrip():
    c = CFG(seed=123, active_modalities=("V", "A"), use_ccc_loss=False)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.yaml")
        c.to_yaml(p)
        c2 = CFG.from_yaml(p)
    assert c2.seed == 123
    assert c2.active_modalities == ("V", "A")
    assert c2.use_ccc_loss is False


def test_update_does_not_mutate():
    c = CFG()
    c2 = c.update(seed=999, fusion_mode="concat")
    assert c.seed == 42 and c.fusion_mode == "weighted"
    assert c2.seed == 999 and c2.fusion_mode == "concat"
