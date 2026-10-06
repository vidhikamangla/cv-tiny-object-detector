# later.md — the curiosity graveyard

Rule: if it is not required by today's milestone deliverable, it goes here.

- [ ] **Distil into the A3 (stride-4) architecture.** A4 showed the student is
      resolution-limited, not supervision-limited, so distillation should be retried where
      the resolution bottleneck is already gone. This is the experiment the A3/A4 pair
      argues for.
- [ ] Feature-mimicking distillation with 1x1 adapters (A4 used logit + box KD only, which
      is the weaker half of what the literature recommends for dense detectors).
- [ ] Train a teacher to convergence rather than stopping it at 20 epochs.
- [ ] Put tiles into the *training* augmentation, rather than bolting tiled inference on at
      test time — the tiling result says the loss comes from a train/inference scale
      mismatch.
- [ ] Copy-paste augmentation for small objects.
- [ ] Varifocal loss row (implemented in `src/losses/cls.py`, never ablated).
- [ ] Higher input resolution (960, 1280) — published VisDrone results use ~1500px.
- [ ] Benchmark A3 rather than A2 in the edge project; A3 is the better model and has a
      more interesting compute profile.
