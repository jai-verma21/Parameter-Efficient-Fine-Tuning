# LLM Fine-Tuning for SQuAD (Llama-2-7B + QLoRA)

**Status:** Plan + results snapshot. The code will be consolidated and cleaned later.  
**Goal:** Efficiently fine-tune **Llama-2-7B** for extractive QA (SQuAD v1.1) under low-VRAM constraints using **QLoRA** (4-bit NF4 adapters), and benchmark accuracy, latency (p95), and memory footprint against lightweight and API baselines.

---

## Why this project
Large models are strong zero-shot but often underperform on extractive QA without task-specific adaptation. QLoRA enables practical fine-tuning on commodity GPUs by quantizing the base model to 4-bit and training small LoRA adapters.

---

## Task & Dataset
- **Task:** Extractive question answering.
- **Dataset:** **SQuAD v1.1** (train/dev) with standard **Exact Match (EM)** and **F1** evaluation.
- **Success criteria:** Raise EM/F1 over the base Llama-2-7B and a TinyLlama baseline while keeping latency and VRAM near base levels.

---

## Method (intended design)
- **Base model:** Llama-2-7B.
- **Adaptation:** **QLoRA** with **4-bit NF4** quantization, LoRA adapters on attention projections (common: q/k/v/o) with moderate rank and α; dropout for stability.
- **Training regime:** supervised fine-tuning on SQuAD v1.1; early stopping on dev EM/F1.
- **Evaluation:** EM/F1 on dev; **p95 latency** and **peak VRAM** for single-question inference.
- **Benchmarks:** 
  1) **TinyLlama-1B**,  
  2) **raw Llama-2-7B** (no fine-tune),  
  3) hosted **Llama-3.1-8B-instant** (API) for reference.

---

## Results snapshot (from the initial run)
- **Fine-tuned Llama-2-7B (QLoRA, 4-bit NF4)** on SQuAD v1.1 improved **EM from 51.3 → 70.3**, with roughly constant latency and memory (**~6 GB VRAM**).
- Outperformed **TinyLlama-1B** by **+41 EM**, and landed **within 9 points** of the hosted **Llama-3.1-8B-instant** reference.
- Achieved **~2× faster local inference** (**p95 ≈ 1.6 s** vs **~3.4 s** via API) with **VRAM comparable** to the base 7B.

> Numbers will be re-validated and expanded with F1, confidence intervals, and ablations once the training/eval harness is finalized.

---

## Planned ablations
- **LoRA rank/target modules** (qkv only vs qkvo).
- **Quantization variants** (NF4 vs FP4) and adapter dropout.
- **Context length and chunking** effects on long questions.
- **Post-training tweaks** (calibration, answer length control).

---

## Limitations
- Extractive QA only; no generative/narrative evaluation yet.
- SQuAD is clean; robustness on noisy domains remains to be tested.
- Results depend on hardware/load; latency comparisons will include methodology.

---

## Acknowledgments
- Meta’s **Llama-2** model family.  
- **SQuAD v1.1** dataset.  
- Prior work on **QLoRA** and 4-bit quantization that enables affordable fine-tuning.
