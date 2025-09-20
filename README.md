# LLM Fine-Tuning for SQuAD (Llama-2-7B + QLoRA)

**Status:** Plan + results snapshot. The notebook is a rough draft and will be cleaned later.

**Goal:** Efficiently fine-tune **Llama-2-7B** for extractive QA (SQuAD v1.1) using **QLoRA** (4-bit NF4) under low-VRAM constraints, and benchmark accuracy (EM), latency (p95), and memory against lightweight and API baselines.

---

## Why
Zero-/few-shot performance on extractive QA is often underwhelming. **QLoRA** allows adapting large models on commodity GPUs by quantizing the base weights to 4-bit and training small LoRA adapters.

---

## Task & Dataset
- **Task:** Extractive question answering  
- **Dataset:** **SQuAD v1.1** (train/dev)  
- **Metric:** **Exact Match (EM)**; same context window and deterministic decoding across models

---

## Intended Method (design, not code)
- **Base:** Llama-2-7B
- **Adaptation:** **QLoRA**, **4-bit NF4**, LoRA **rank=16, alpha=16**
- **Targets:** attention + MLP projections; ~**0.2%** of weights trainable
- **Seq length:** **512**
- **Training:** supervised fine-tuning on SQuAD v1.1 with early stopping
- **Eval:** EM on dev; **p95 latency** and **peak VRAM** for single-query inference
- **Baselines:** TinyLlama-1B, raw Llama-2-7B, and a hosted Llama-3.1-8B-instant (API) reference

---

## Results Snapshot

- Built a **SQuAD v1.1 harness** with a fixed prompt and **deterministic decoding**; scored EM under the same window for every model.
- **Fine-tuned Llama-2-7B (QLoRA, 4-bit NF4)** on Colab T4; adapters on attention+MLP; `rank=16`, `alpha=16`; **~0.2%** trainable params; **seq_len=512**.
- **Fair comparison** across three local systems:

  | Model                           | EM |
  |---------------------------------|---:|
  | TinyLlama-SQuAD (1B)            | **29.33** |
  | Llama-2-7B (raw, no FT)         | **54.33** |
  | Llama-2-7B **QLoRA** (this run) | **70.33** |

- Hosted reference: **Llama-3.1-8B-instant (API)** at **EM 79.33**.
- **Latency & Memory:** local QLoRA-7B was **~2× faster** to query (**p95 ≈ 1.6 s**) than the API (**~3.4 s p95**), with **~6 GB VRAM**—comparable to raw 7B.

> Numbers are from an initial run; they’ll be re-validated with F1, seeds, and ablations once the training/eval harness is finalized.

---

## Planned Ablations
- LoRA **rank/targets** (qkv vs qkvo), adapter dropout
- **NF4 vs FP4** quantization
- Context length effects and chunking
- Post-training calibration and answer-length control

---

## Limitations
- Focused on extractive QA; robustness on noisy domains not yet tested
- SQuAD is clean and short-context; long-context behavior not evaluated
- Latency depends on hardware and batch size; methodology will be published with the cleaned code

---

## Acknowledgments
- Meta **Llama-2** family  
- **SQuAD v1.1** dataset  
- QLoRA and 4-bit quantization research enabling affordable fine-tuning
