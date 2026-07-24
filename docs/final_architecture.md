# Final Architecture

The repository implements duration-based physiological data processing, subject-aware benchmarks, fast current-state detection, slow multi-horizon forecasting, and a replayable supervisory policy architecture.

```mermaid
flowchart TD
  A[Dataset Ingestion] --> B[Target Construction]
  B --> C[Subject Split]
  C --> D[Calibration and Normalization]
  D --> E[Fast Detector 0.25s cadence]
  D --> F[Slow Forecaster 1s cadence, 5s/30s horizons]
  G[Timestamped Physical Inputs 0.05s cadence] --> H[Deterministic Physical Kernel]
  H --> I[State Machine]
  E --> I
  F --> I
  I --> J[Supervisory Actions]
  J --> K[Replay Audit Artifacts]
```

Physiological ML is outside any certified emergency-stop chain. Physical emergency and protective rules have higher authority than ML layers. The physical kernel is configurable deterministic logic inspired by separation-monitoring and stopping-distance principles; it is not an ISO-compliant implementation.

Artifact provenance is saved through config hashes, artifact paths, artifact hashes, split subjects, target definitions, timing settings, and model version fields where prediction artifacts are available.

