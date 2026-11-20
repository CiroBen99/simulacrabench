import time
STARTED=time.perf_counter()
import json
from pathlib import Path
import sdm
from simulacra.kumo_only_runtime import KumoOnly
from simulacra.kumo_large_only_runtime import FrozenKumoLarge
ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config.json').read_text())
PREDICTOR=KumoOnly(CONFIG,ROOT,started=STARTED,engine_factory=lambda:FrozenKumoLarge(ROOT,CONFIG))
def predict(frame,schema):
    # A call-local slice bound admits every row; no fixed donor cap remains.
    # The original predictor still validates masks and selects COMPLETE rows.
    PREDICTOR.config=dict(CONFIG,support_caps={},default_support_cap=len(frame))
    return PREDICTOR.predict(frame,schema)
