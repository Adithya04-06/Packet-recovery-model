# iTantra - ML-based Text Recovery Prototype

## Problem Statement & Motivation
iTantra is an offline, low-bandwidth multilingual communication system. In wireless environments (Bluetooth, Wi-Fi, LoRa), packet loss is a frequent issue. Traditional retransmission might not be viable in highly constrained, offline edge scenarios. 

This module provides a sequence-to-sequence ML model that mathematically "guesses" and reconstructs missing chunks of text caused by dropped packets, leveraging the semantic context of the successfully received packets.

## Important Conceptual Distinction
**PACKET RECOVERY vs. ML TEXT RECOVERY**
The system identifies *which* packets are missing using Sequence Numbers, CRC checksums, and a packet reassembler. Cryptography (AES-GCM) is handled strictly before and after the channel simulation. The ML model *does not* reconstruct arbitrary encrypted bytes or recover encryption keys; rather, it reconstructs the logical text that was encapsulated in the lost payloads. Missing data is conceptually passed to the model as `<MISSING>`.

## Architecture Flow
```text
Sender -> Text -> Encoding -> Encryption -> Packetization 
-> Wireless Channel (Packet Loss) -> Receiver
-> Packet Reassembly (Detects Missing Seqs) -> Decryption
-> <MISSING> placeholder insertion -> ML Text Recovery Model
-> Recovered Text
```

## Features
- **Cryptography Pipeline**: Per-payload AES-GCM authenticated encryption.
- **Packet Loss Simulation**: Configurable uniform/burst loss rates.
- **Offline ML Model**: Uses HuggingFace `t5-small` (local, no internet/API calls required) to perform Seq2Seq reconstruction.
- **Evaluation**: Calculates Exact Match, Word-Level Accuracy, BLEU, and ROUGE-L.
- **Demo Interface**: A local Gradio web app for visualization.

## Setup Instructions

### 1. Installation
This project requires Python 3.9+. Create a virtual environment and install the dependencies:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Generation
Generates synthetic corrupted sentences from clean English text for training/testing.
```bash
python -m recovery.dataset_generator
```

### 3. Training
Fine-tune the T5 model on the generated corrupted datasets.
```bash
python -m recovery.train
```

### 4. Evaluation Experiments
Run the pipeline at various packet loss rates (0%, 5%, 10%, 20%, 30%, 40%) to generate accuracy metrics and a comparison graph.
```bash
python -m evaluation.experiments
```
*(This generates `evaluation_graph.png`)*

### 5. Launch Demo Application
Launch the local Gradio interface to manually interact with the pipeline.
```bash
python -m app.demo_interface
```

## Limitations and Future Improvements
- **Multilingual Support**: Currently configured for English (`t5-small`). Can be swapped to `google/mt5-small` to support Indian languages (Hindi, Tamil, etc.).
- **Model Size**: The current model is full FP32. For edge deployment, the model should be quantized (INT8/FP16) or converted to ONNX.
- **Character Splitting**: UTF-8 multibyte characters split across packet boundaries are handled aggressively via `errors='ignore'`. Future versions could employ forward-error-correction (FEC) or character-aware chunking prior to encryption.
