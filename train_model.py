"""Train the SPACE anomaly model on real TESS light curves and save checkpoint_v2.pt.

Usage:  python train_model.py
Needs internet access to MAST. If fewer than 3 stars download, it falls back to a
synthetic training set and says so.
"""
from core import train as T

per = T.build_real_dataset(sectors_per_target=2)
if len(per) >= 3:
    model, thr, meta = T.train_pipeline(per, epochs=80, source=f"real TESS ({len(per)} stars)")
else:
    print("Fewer than 3 real stars downloaded; falling back to SYNTHETIC training.")
    per = T.build_synthetic_dataset(40)
    model, thr, meta = T.train_pipeline(per, epochs=60, source="synthetic fallback")
