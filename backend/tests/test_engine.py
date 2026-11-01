"""Engine telemetry — the step frames the studio's charts are drawn from.

    cd backend && python -m pytest -q tests/test_engine.py
"""
from engine import TrainingEngine


class Frames:
    def __init__(self):
        self.sent = []

    def publish(self, frame):
        self.sent.append(frame)


def test_step_frames_describe_the_training_batch():
    out = Frames()
    eng = TrainingEngine(out)
    eng.create({"name": "t", "template": "text", "size": "tiny",
                "data": {"source": "paste", "text": "to be, or not to be, that is the question. " * 40},
                "tokenizer": {"kind": "char", "vocab_size": 256},
                "chunking": {"block_size": 64, "strategy": "window"},
                "train": {"grad_clip": 0.05}})
    for _ in range(9):
        eng.training_step()
    steps = [f for f in out.sent if f.get("type") == "step"]
    assert len(steps) == 9
    # steps 0 and 8 also sample a preview; its forward passes must not replace the batch's attention
    assert {len(f["attention"]["layers"][0]["heads"][0]["matrix"]) for f in steps} == {64}
    # the gradient norm is what backprop produced, not flattened to the clip value
    assert max(f["grad_norm"] for f in steps) > 0.1
