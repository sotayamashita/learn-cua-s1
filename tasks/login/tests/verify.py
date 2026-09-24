"""Grade the login choice; labels are only available to the verifier."""

import json
from pathlib import Path

prediction = json.loads(Path("/app/prediction.json").read_text())
selected = prediction["selected"]
if selected not in {"login", "login-google"}:
    raise ValueError(f"Unknown candidate: {selected}")
Path("/logs/verifier/reward.txt").write_text(str(int(selected == "login")))
