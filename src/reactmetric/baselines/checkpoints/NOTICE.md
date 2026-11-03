# Pretrained GRU checkpoints

The GRU baselines reported in Breseman and Woollard, *REACT: Measuring Regime Adaptation in
Trajectories* (TGF 2026), trained on Eindhoven Centraal pedestrian trajectories at 10 Hz.

- `gru_position_best.pt`: position-history GRU (input: normalized x, y)
- `gru_orientation_best.pt`: orientation-augmented GRU (input: x, y and sin/cos heading)

Each file holds the network weights and its training settings only (`torch.load(...,
weights_only=True)`); `config` records the history (30 observations), horizon (15), split seed
and validation ADE/FDE.

Training data: Pouw, C. A. S., Corbetta, A., Gabbana, A., van der Laan, C., Toschi, F.,
"Data on high-resolution pedestrian trajectories at Eindhoven Centraal train station",
Zenodo, licensed CC-BY-4.0. The weights are derived from that data; no track data is included
in these files.

Load via `rm.baselines.GRUPosition()` and `rm.baselines.GRUOrientation()`.
