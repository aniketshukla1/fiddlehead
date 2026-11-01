"""Saved-model folders (saved/<name>/) each carry a JSON sidecar describing the model.

It is written as fiddlehead.json. Folders saved before the project was renamed hold
signalscope.json instead; both are read, so old checkpoints keep loading.
"""
import os

# where saved models live; FIDDLEHEAD_SAVE_DIR moves them (a volume, a bigger disk)
SAVE_DIR = os.environ.get("FIDDLEHEAD_SAVE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")

META = "fiddlehead.json"
LEGACY_META = "signalscope.json"


def meta_path(d: str) -> str:
    """The sidecar to read in folder `d`: the current name, else the legacy one."""
    p = os.path.join(d, META)
    if os.path.isfile(p):
        return p
    legacy = os.path.join(d, LEGACY_META)
    return legacy if os.path.isfile(legacy) else p


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        assert meta_path(d).endswith(META)                      # nothing yet → where to write
        open(os.path.join(d, LEGACY_META), "w").close()
        assert meta_path(d).endswith(LEGACY_META)               # old folder → legacy file
        open(os.path.join(d, META), "w").close()
        assert meta_path(d).endswith(META)                      # re-saved → current wins
    print("artifacts ok")
