# Project AETHER: xLAM-AETHER-v1 Model Card & Technical Report

## 1. Executive Summary
**xLAM-AETHER-v1** is a specialized, 3-Billion parameter Large Language Model (LLM) engineered to serve as the cognitive engine for autonomous agentic workflows. 

Developed as the core component of **Project AETHER**, this model is designed to act not just as a conversational chatbot, but as a **ReAct (Reasoning + Acting) agent**. It has been fine-tuned to understand tool schemas, reason through a user's goal, and output highly structured, syntactically perfect JSON payloads to trigger external tools (such as shell commands or API calls).

Built upon the robust `Salesforce/xLAM-2-3b-fc-r` architecture (which is based on Qwen2.5), the model was distilled using QLoRA and quantized to a highly efficient 4-bit GGUF format. This allows it to run locally on consumer-grade laptops while retaining the complex reasoning capabilities of its parent architecture.

## 2. Core Capabilities
- **Dual-Mode Operation:** The model seamlessly switches between general conversational chat (natural language) and Agentic execution (strict JSON tool calls) without suffering from catastrophic forgetting.
- **High-Fidelity Tool Calling:** Outputs perfectly formatted JSON arrays exactly as requested in the system prompt, ready for immediate parsing by Python/Rust backends.
- **Context Window:** 32,768 tokens (native), allowing for long multi-turn agent loops, massive tool schemas, and extensive observation histories.
- **Low Hardware Footprint:** Quantized to `Q4_K_M` (~1.93 GB), allowing it to run entirely in system RAM on standard laptops or free cloud tiers.

## 3. Technical Specifications

| Attribute | Detail |
| :--- | :--- |
| **Base Model** | `Salesforce/xLAM-2-3b-fc-r` (Qwen2 architecture) |
| **Total Parameters** | 3.09 Billion (Base) + 7.3 Million (LoRA Adapter) |
| **Fine-Tuning Method** | QLoRA (Quantized Low-Rank Adaptation) |
| **Training Precision** | 4-bit NormalFloat (NF4) with FP16 compute |
| **Dataset** | `Salesforce/APIGen-MT-5k` (5,000 verified multi-turn function-calling trajectories) |
| **Training Hardware** | Google Colab (NVIDIA Tesla T4) |
| **Final Format** | GGUF (`q4_k_m` quantization) |
| **Chat Template** | ChatML (`<\|im_start\|>` / `<\|im_end\|>`) |
| **File Size** | 1.93 GB |
| **License** | CC-BY-NC-4.0 (Non-Commercial) |

## 4. End-to-End Training Pipeline

### 4.1 Data Synthesis & Curation
Rather than training from scratch, the model utilized the `Salesforce/APIGen-MT-5k` dataset. This dataset contains strictly verified, multi-turn API interactions. The data was formatted using the official ChatML template, ensuring the model learned the exact token boundaries for when to start reasoning and when to trigger a tool.

### 4.2 The Distillation Protocol (QLoRA)
To inject reasoning capabilities into the 3B model without requiring enterprise-grade GPUs, the training utilized QLoRA:
- **Base Quantization:** The base model was loaded in 4-bit NF4.
- **Adapter Injection:** Trainable low-rank matrices (Matrix A and B) were injected into the attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
- **Hyperparameters:** Rank (\(r\)) was set to 16, Alpha (\(\alpha\)) to 32, with a learning rate of \(2e-4\).
- **Efficiency:** Using the **Unsloth** framework, the training loop was optimized for the T4 GPU. The model was trained for 1 epoch (625 steps), achieving a final training loss of `0.0003`, indicating high convergence on the target ReAct formatting.

### 4.3 Merging & Quantization
Post-training, the 28MB LoRA adapter was mathematically merged with the 6GB base model. To ensure the model could run on edge devices, the merged weights were converted to GGUF format and quantized to **Q4_K_M**. This 4-bit integer quantization protects the "salient" weights (those contributing most to the reasoning spikes) while compressing the model to under 2GB.

## 5. Benchmark & Behavior Tests

Post-deployment, the model was subjected to a series of behavioral tests to ensure it did not lose its base intelligence while learning the tool-calling format.

**Test A: General Knowledge (Chat Mode)**
- *Prompt:* "What are the cultural beliefs of China and Egypt? Give a short summary."
- *Output:* Coherent, well-structured natural language summary covering mummification, Confucianism, and filial piety. No hallucinated JSON. Retained its "chat" persona.

**Test B: Reasoning (Math Mode)**
- *Prompt:* "If a train travels at 60 mph for 2.5 hours, how far does it go? Explain the math."
- *Output:* Step-by-step breakdown using LaTeX formatting, correctly applying the formula \(Distance = Speed \times Time\), and concluding with `150 miles`.

**Test C: Agentic Tool Calling (ReAct Mode)**
- *Prompt:* *"List all .tmp files in /tmp/aether-sandbox."* (Provided with a `shell` tool schema).
- *Output:* `[{"name": "shell", "arguments": {"cmd": "find /tmp/aether-sandbox -type f -name '.tmp'"}}]`
- *Latency:* 1.12 seconds (via API on a Colab T4 GPU).

## 6. Deployment & Inference

### Option 1: Local Inference (Ollama / LM Studio)
Due to the GGUF format, the model can be run locally. A `Modelfile` is required to enforce the ChatML template.
```text
FROM ./xlam-aether-q4_k_m.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.1
```

### Option 2: OpenAI-Compatible API (Colab)
Using `llama-cpp-python`, the model can be served as a local OpenAI-compatible API. This is ideal for building FastAPI backends that interact with the model.
```bash
python -m llama_cpp.server --model xlam-aether-q4_k_m.gguf --n_gpu_layers 99 --chat_format chatml --port 8000
```

## 7. Limitations & Future Work

- **License Constraints:** Because it inherits from the xLAM base model, this model is licensed under **CC-BY-NC-4.0**. It is strictly for research and personal use. Commercial deployment would require re-training on a permissive base model (e.g., Llama-3.2-3B).
- **Hardware Bottlenecks:** While fast on a GPU (~1.1s response), running this model purely on a CPU (like Hugging Face's free tier) results in massive latency (~250 seconds per prompt) due to the prefill phase processing the tool schema.
- **Parallel Tool Calling:** The model currently excels at single-action tool calls. Future training iterations will focus on emitting arrays of multiple parallel tool calls.

## 8. Links & Resources
- **Model Repository:** [huggingface.co/iamrahull/xlam-ather](https://huggingface.co/iamrahull/xlam-ather)
- **Demo Space:** [huggingface.co/spaces/iamrahull/xlam-ather](https://huggingface.co/spaces/iamrahull/xlam-ather)
- **Training Dataset:** [Salesforce/APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)

---
*Report generated for Project AETHER. Designed and trained by Rahul.*
---
license: cc-by-nc-4.0
---
