# Pretrained GRU checkpoints

These weights were trained for the original REACT research on a large corpus of
retail pedestrian motion (Standard AI / Standard Cognition tracks-style data).

Files:
- gru_position_best.pt — position-history GRU (input: normalized x,y)
- gru_orientation_best.pt — orientation-augmented GRU (input: x,y + sin/cos heading)

Load via:
  rm.baselines.GRUPosition()
  rm.baselines.GRUOrientation()
