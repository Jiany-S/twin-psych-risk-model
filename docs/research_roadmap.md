# Research Roadmap

The next real experiment should be a controlled 3D concrete printing or HRC study designed for synchronized physiological, physical, and intervention data collection. No such data collection has occurred in this repository.

## Required Channels

Synchronize worker physiology, worker position and velocity, robot position and velocity, robot state, stopping distance, task phase, worker role, sensor quality, process state, intervention timestamps, near-miss annotations, and operator reports.

## Protocol

Use a pre-task resting calibration segment, controlled non-hazard scenarios, carefully supervised approach/separation scenarios, task-boundary pauses, and explicit operator annotations. Preserve participant-level train/validation/test splits.

## Ethics And Safety

Scenarios must be designed so deterministic physical safety systems remain authoritative. Physiological ML should be advisory during data collection. Participants need informed consent, stop authority, privacy protections, and clear exclusion criteria.

## Evaluation

Report model calibration, AUROC/AUPRC, false warnings per hour, warning lead time by annotated event, missed near-miss periods, operator response time, state-machine stability, and deterministic safety-rule activations. Do not use protocol stress labels as collision-risk labels.

