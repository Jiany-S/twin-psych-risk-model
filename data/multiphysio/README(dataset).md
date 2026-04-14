# MultiPhysio-HRC Dataset

**MultiPhysio-HRC** is a multimodal dataset collected to study psycho-physiological states in industrial Human-Robot Collaboration (HRC) scenarios. It includes physiological, audio, and facial data recorded during tasks designed to elicit different levels of stress, cognitive load, and emotional states.

### Contents

The dataset provides synchronized recordings of:
- Electroencephalography (EEG) – 12-channel dry EEG
- Electrocardiography (ECG)
- Electrodermal Activity (EDA)
- Respiration (RESP)
- Electromyography (EMG) – trapezius muscle

The following features are also included:
- Voice features – during rest, tasks, and HRC
- Facial Action Units (AUs) – extracted from video
- Ground truth questionnaires – STAI-Y1, NASA-TLX, SAM, NARS

### Experimental Protocol

The dataset was collected over two sessions:
- Day 1 – Baseline and Stress Induction
    - Rest
    - Cognitive tasks (Stroop easy, Stroop hard, N-back, Arithmetic, Hanoi Tower)
    - Breathing exercise
    - Virtual Reality task (Richie’s Plank Experience)

- Day 2 – Manual and Robot-Assisted Disassembly
    - Rest
    - Manual battery disassembly
    - Collaborative disassembly with a Fanuc CRX-20 cobot, using voice commands

Each task was followed by self-report questionnaires to provide ground truth labels.

### Participants

- 52 participants in Day 1
- 42 participants continued in Day 2
- Age: 27.98 ± 10.22 years
- Gender: 48 male, 7 female

### Acquisition Devices

- EEG: [Bitbrain Diadem](https://www.bitbrain.com/neurotechnology-products/dry-eeg/diadem) (12-channel dry EEG; AF7, Fp1, Fp2, AF8, F3, F4, P3, P4, PO7, O1, O2, PO8; ground and reference on left earlobe).
- ECG, EDA, RESP, EMG: [Bitbrain Versatile Bio sensor](https://www.bitbrain.com/neurotechnology-products/biosignals/versatile-bio).
    - ECG: V2 placement.
    - EDA: index and middle fingers of non-dominant hand.   
    - EMG: right trapezius muscle.
    - Respiration: chest band.
- Audio: Bluetooth microphone.
- Video: Standard webcam (frontal view).

Sampling rate: 256 Hz for physiological signals (synchronized with audio/video via [SennsLab software](https://www.bitbrain.com/neurotechnology-products/software/sennslab)).

### Data Processing and Features

All physiological signals were recorded at 256 Hz and synchronized with audio-video.

Pre-processing pipelines for EEG, ECG, EMG, EDA, and RESP included filtering and artifact removal.

Features extracted:

- ECG/HRV: time, frequency, and nonlinear measures
- EMG: time and spectral features
- EDA: tonic and phasic decomposition, peaks
- RESP: rate variability and spectral indices
- EEG: PSD in δ, θ, α, β, γ bands, entropy measures, hemispheric ratios
- Voice: MFCCs, shimmer, jitter, prosodic statistics
- Text: transcriptions (Whisper) + sentence embeddings

### Data Structure
```
MultiPhysio-HRC/
│
├── physiological_data/
│   ├── filtered/                # Preprocessed signals
│   │   ├── subj1/
│   │   │   ├── task1.csv
│   │   │   ├── task2.csv
│   │   │   ...
│   │   └── subj2/
│   │       ├── task1.csv
│   │       ├── task2.csv
│   │       ...
│   │
│   └── raw/                     # Raw signals as acquired
│       ├── subj1/
│       │   ├── task1.csv
│       │   ├── task2.csv
│       │   ...
│       └── subj2/
│           ├── task1.csv
│           ├── task2.csv
│           ...
│
├── features/                    # Extracted features and labels
│   ├── aus_data.csv
│   ├── bio_features_60s.csv
│   ├── eeg_features_5s.csv
│   ├── nlp_embeddings.csv
│   ├── speech_features.csv
│   └── labels.csv
|
├── participants_task_overview.csv
├── features_table.pdf  
└── README.md
```

- Raw physiological data: direct sensor recordings (EEG, ECG, EDA, EMG, RESP).
- Filtered physiological data: preprocessed signals (artifact removal, filtering, down-sampling).
- Features: aggregated files containing feature vectors for all modalities + questionnaire-based labels.
- Participant overview of the performed tasks.

### Applications

This dataset supports research in:

- Mental state recognition (stress, cognitive load, emotional dimensions)
- Multimodal machine learning and sensor fusion
- Affective computing and human-aware robotics
- Workplace ergonomics and well-being in Industry 5.0

### Ethics and Consent

Approved by the Ethics Committee of the University of Applied Sciences and Arts of Southern Switzerland (SUPSI).

Informed consent was obtained from all participants. Data are pseudonymized.

### FAQ

**Q: How do I get access to raw videos or audio logs?**

Please send and email with your request.

### Citation

Information on how to cite the dataset and the paper are provided in [https://automation-robotics-machines.github.io/MultiPhysio-HRC.github.io/](https://automation-robotics-machines.github.io/MultiPhysio-HRC.github.io/)


### Contact

Lead contact: andrea.bussolan@supsi.ch 

Issues & questions: please open a GitHub issue.
