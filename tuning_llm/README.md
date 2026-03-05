# Fine-Tuning LLMs with PEFT 🎯

A comprehensive guide to fine-tuning large language models using Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically QLoRA (Quantized Low-Rank Adaptation), enabling efficient model customization on consumer GPUs.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Technologies](#-key-technologies)
- [Architecture](#-architecture)
- [Notebook Contents](#notebook-contents)
  - [Load a 4-bit Model (QLoRA)](#load-a-4-bit-model-qlora)
  - [Add LoRA Adapters](#add-lora-adapters)
  - [Train with QLoRA](#train-with-qlora)
  - [Test the Fine-Tuned Model](#test-the-fine-tuned-model)
  - [Save LoRA Adapter](#save-lora-adapter)
  - [Architecture Overview](#architecture-overview)
  - [DPO Alignment](#dpo-direct-preference-optimization-alignment)
- [Setup](#-setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)

---

## 🎯 Overview

This project demonstrates how to efficiently fine-tune large language models using PEFT techniques, specifically QLoRA, which enables:

✅ **Memory Efficiency** - Train models on consumer GPUs using 4-bit quantization

✅ **Cost Effectiveness** - Reduce training costs by ~90% compared to full fine-tuning

✅ **Parameter Efficiency** - Only train small adapter layers instead of entire model

✅ **Domain Adaptation** - Customize models for specific tasks and domains

✅ **Model Preservation** - Keep base model weights unchanged, only add adapters

**Use Cases:**
- Domain-specific chatbots (medical, legal, technical support)
- Custom instruction-following models
- Task-specific language models
- Hybrid RAG + fine-tuning systems
- Cost-effective model customization for startups

---

## 🛠️ Key Technologies

| Technology | Purpose |
|------------|---------|
| **PEFT (Parameter-Efficient Fine-Tuning)** | Framework for efficient model adaptation |
| **QLoRA (Quantized LoRA)** | 4-bit quantization + Low-Rank Adaptation |
| **LoRA (Low-Rank Adaptation)** | Trainable adapter layers without modifying base model |
| **BitsAndBytes** | 4-bit quantization library |
| **Transformers** | Hugging Face transformers library |
| **Mistral 7B** | Base language model (can be replaced with other models) |
| **DPO (Direct Preference Optimization)** | Alignment technique for human preferences |
| **PyTorch** | Deep learning framework |

---

## 🏗️ Architecture

### What is Quantization?

**Quantization** is a technique that reduces the precision of model weights to use less memory and compute.

#### Understanding Precision Levels

| Precision | Bits per Weight | Memory (7B Model) | Example Values |
|-----------|----------------|-------------------|----------------|
| **FP32** (Full Precision) | 32 bits | ~28GB | 0.12345678901234567890 |
| **FP16** (Half Precision) | 16 bits | ~14GB | 0.12346 |
| **BF16** (Brain Float) | 16 bits | ~14GB | 0.12346 |
| **INT8** (8-bit) | 8 bits | ~7GB | 0.123 |
| **INT4** (4-bit) | 4 bits | ~4GB | 0.12 |

**How Quantization Works:**

```
Original Weight (FP16):  0.123456789
         ↓
Quantization Process
         ↓
Quantized Weight (4-bit): 0.12
```

**Key Insight:** Instead of storing weights as 16-bit floating-point numbers, quantization stores them as 4-bit integers, reducing memory by 4× with minimal accuracy loss.

#### Why Quantization Works

1. **Neural networks are robust** - Small precision losses don't significantly impact performance
2. **Weights cluster** - Most weights are near zero, so fewer bits can represent them efficiently
3. **Activation quantization** - Can quantize activations during inference for even more savings
4. **Modern techniques** - NF4 quantization format is optimized for neural networks

#### Types of Quantization

**Post-Training Quantization (PTQ):**
- Quantize model after training
- Fast and easy
- Used in QLoRA (quantize base model, train adapters)

**Quantization-Aware Training (QAT):**
- Train model with quantization in mind
- Better accuracy but slower
- Used in some specialized scenarios

**4-bit Quantization (NF4):**
- Special format optimized for neural networks
- Used in QLoRA
- Provides best balance of compression and accuracy

#### Memory Savings Example

For a 7B parameter model:

```
FP16 Model:     7B × 2 bytes = 14GB
4-bit Model:    7B × 0.5 bytes = 3.5GB

Savings: 75% reduction (14GB → 3.5GB)
```

**This is why QLoRA can fit on a single T4 GPU (16GB) instead of requiring multiple A100 GPUs!**

---

### QLoRA Fine-Tuning Architecture

```
┌─────────────────────┐
│  Base Model         │
│  (Mistral 7B)       │
│  ~13GB (FP16)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4-bit Quantization │
│  ~4GB (4-bit)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  LoRA Adapters      │
│  Trainable layers   │
│  ~100MB             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Fine-Tuned Model   │
│  Domain-specific    │
└─────────────────────┘
```

### Hybrid RAG + Fine-Tuning System

```
User Question
     │
     ▼
┌─────────────┐
│ RAG         │ ← Retrieve relevant context
│ Retrieval   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Fine-tuned  │ ← Generate domain-specific answer
│ LLM         │   using retrieved context
└──────┬──────┘
       │
       ▼
Better Domain Answer
```

### Memory Comparison

| Method | Cost | GPU Requirements | Memory Usage |
|--------|------|------------------|--------------|
| **Full Fine-Tuning** | $$$$ | Many GPUs | ~40GB+ (FP16) |
| **QLoRA** | $ | 1 GPU | ~4GB (4-bit) + adapters |

**Why QLoRA Works:**
- Base model = 4-bit quantized (frozen, not trainable)
- Only LoRA adapters train (small, efficient)
- Reduces training memory by ~90%
- Enables fine-tuning on consumer GPUs (T4, V100, etc.)

---

## 📝 Notebook Contents

### **Load a 4-bit Model (QLoRA)**

**Objective:** Load a quantized model that fits into small GPUs for efficient fine-tuning.

#### Key Concepts

1. **4-bit Quantization** - Reduce model precision from FP16 to 4-bit integers (see [Quantization explanation](#what-is-quantization) above)
2. **BitsAndBytes Integration** - Use BitsAndBytes library for efficient quantization
3. **Model Loading** - Load base model with quantization config
4. **Memory Optimization** - Enable efficient memory usage for training
5. **NF4 Format** - Neural network-optimized 4-bit quantization format

---

#### Implementation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
```

**Key Benefits:**
- ✅ Model size reduced from ~13GB to ~4GB
- ✅ Fits on single GPU (T4, V100, etc.)
- ✅ Base model weights frozen (not trainable)
- ✅ Only adapters will be trained

---

### **Add LoRA Adapters**

**Objective:** Add trainable adapter layers without modifying the full model weights.

#### Why Do We Need LoRA Adapters?

**Critical Question:** If we've quantized the model, why do we need LoRA adapters?

**Answer:** The quantized base model is **frozen** (not trainable). LoRA adapters are the **only trainable components** that allow fine-tuning.

**How QLoRA Works:**

```
┌─────────────────────────────────┐
│  Base Model (Mistral 7B)        │
│  ↓                              │
│  4-bit Quantization              │
│  ↓                              │
│  FROZEN (Not Trainable) ❄️      │ ← Can't modify these weights
└─────────────────────────────────┘
           +
┌─────────────────────────────────┐
│  LoRA Adapters                  │
│  Small trainable matrices       │
│  TRAINABLE ✅                   │ ← Only these get updated!
└─────────────────────────────────┘
           =
┌─────────────────────────────────┐
│  Fine-Tuned Model               │
│  (Base frozen + Adapters trained)│
└─────────────────────────────────┘
```

**Why Not Train the Quantized Model Directly?**

1. **Memory Constraints** - Even quantized, training all 7B parameters requires too much memory
2. **Gradient Storage** - Need to store gradients for all parameters during training
3. **Efficiency** - LoRA adapters are tiny (~100MB) vs full model (3.5GB)
4. **Modularity** - Can swap adapters for different tasks without retraining base model

**The Magic of LoRA:**

Instead of updating all 7B parameters:
- **Full fine-tuning**: Update 7,000,000,000 parameters ❌
- **LoRA**: Update only ~10,000,000 parameters ✅ (0.1% of model)

**LoRA adapters learn to "modify" the base model's behavior** by adding small adjustments to attention layers, without changing the base weights themselves.

---

#### Key Concepts

1. **LoRA (Low-Rank Adaptation)** - Add small trainable matrices to attention layers
2. **Parameter Efficiency** - Only train ~0.1-1% of model parameters
3. **Adapter Configuration** - Configure rank, alpha, and target modules
4. **PEFT Integration** - Use PEFT library for LoRA implementation
5. **Frozen Base Model** - Base model weights stay unchanged, only adapters train

---

#### Implementation

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank (lower = fewer parameters)
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

#### Why These Target Modules?

**Understanding Transformer Attention:**

Transformer models use **self-attention** to understand relationships between tokens. The attention mechanism has four key projection matrices:

```
Input Tokens
     │
     ├─→ Q (Query)   ← "What am I looking for?"
     ├─→ K (Key)     ← "What do I represent?"
     ├─→ V (Value)   ← "What information do I contain?"
     │
     ▼
Attention Scores = Q × K^T  (How much to attend to each token)
     │
     ▼
Output = Attention × V      (Weighted combination)
     │
     ▼
O (Output Projection)        ← Final transformation
```

**Why Target These Modules?**

1. **Q, K, V Projections (`q_proj`, `k_proj`, `v_proj`)**:
   - Control **what** the model attends to
   - Determine **which tokens** are important
   - Most critical for adapting model behavior
   - ~75% of attention parameters

2. **O Projection (`o_proj`)**:
   - Controls **how** attended information is combined
   - Final transformation before output
   - Important for output formatting
   - ~25% of attention parameters

**Why Not Other Layers?**

| Layer Type | Why Usually Not Targeted | When to Target |
|------------|-------------------------|----------------|
| **Embedding** | Rarely needs adaptation | Domain-specific vocabularies |
| **MLP/FFN** | Less critical for adaptation | Complex reasoning tasks |
| **Layer Norm** | Very small, limited impact | Fine-grained control |
| **Output Head** | Task-specific, often retrained separately | Classification tasks |

**Research Findings:**

- **Attention layers** account for ~40% of parameters but ~80% of adaptation benefit
- Targeting Q, K, V, O gives best performance/efficiency trade-off
- Adding MLP layers adds ~20% more parameters but only ~5% performance gain

**Alternative Configurations:**

```python
# Minimal (fastest, least parameters)
target_modules=["q_proj", "v_proj"]  # Only Q and V

# Standard (recommended) ⭐
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# Extended (better performance, more parameters)
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# All Linear Layers (maximum adaptation, most parameters)
target_modules="all-linear"  # Targets all linear layers
```

**Best Practice:**

Start with `["q_proj", "k_proj", "v_proj", "o_proj"]` (standard configuration):
- ✅ Best balance of performance and efficiency
- ✅ Works well for most tasks
- ✅ Well-tested and documented
- ✅ ~0.25% of parameters trainable

Only expand if:
- You need maximum performance and have resources
- Standard config doesn't meet your requirements
- You're fine-tuning for very complex reasoning tasks

# Prepare model for training
```python
model = prepare_model_for_kbit_training(model)
```
# Add LoRA adapters
```python
model = get_peft_model(model, lora_config)
```
# Print trainable parameters
```python
model.print_trainable_parameters()
```

**Output Example:**
```python
trainable params: 8,388,608 || all params: 3,411,550,208 || trainable%: 0.25
```

**What This Means:**
- **8.4M parameters** are trainable (the LoRA adapters)
- **3.4B parameters** are frozen (the quantized base model)
- Only **0.25%** of the model gets updated during training!

**How LoRA Adapters Work:**

LoRA doesn't modify the base model weights. Instead, it adds small matrices that get **multiplied** with the base weights during forward pass:

```
Original: output = W × input
With LoRA: output = (W + ΔW) × input
           where ΔW = B × A (low-rank decomposition)
```

- **W** = Frozen base model weights (quantized)
- **A, B** = Small trainable LoRA matrices (only these train!)
- **ΔW** = Learned adaptation (B × A)

**Key Benefits:**
- ✅ Only ~0.25% of parameters are trainable
- ✅ Adapters can be saved/loaded independently (~100MB vs 3.5GB)
- ✅ Multiple adapters can be swapped for different tasks
- ✅ Base model remains unchanged (can reuse for other tasks)
- ✅ Memory efficient (only store gradients for adapters)

---

### **Train with QLoRA**

**Objective:** Train the LoRA adapters on your custom dataset.

#### Key Concepts

1. **Training Configuration** - Set learning rate, batch size, epochs
2. **Data Preparation** - Format dataset for instruction-following
3. **Training Loop** - Fine-tune adapters while base model stays frozen
4. **Memory Management** - Optimize for GPU memory constraints

---

#### Implementation

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

# Prepare training data
def format_instruction(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

dataset = Dataset.from_list(your_data)
dataset = dataset.map(format_instruction)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Train
trainer.train()
```

**Training Tips:**
- ✅ Use gradient accumulation for effective larger batch sizes
- ✅ Use `paged_adamw_8bit` optimizer for memory efficiency
- ✅ Monitor GPU memory usage
- ✅ Save checkpoints regularly

---

### **Test the Fine-Tuned Model**

**Objective:** Evaluate the fine-tuned model's performance on test examples.

#### Key Concepts

1. **Inference Setup** - Load model and adapters for inference
2. **Text Generation** - Generate responses using fine-tuned model
3. **Comparison** - Compare base model vs. fine-tuned model outputs
4. **Evaluation Metrics** - Assess quality and domain-specific performance

---

#### Implementation

```python
# Load fine-tuned model
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "./results/checkpoint-1000")

# Generate response
def generate_response(instruction):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

# Test
instruction = "Explain quantum computing in simple terms"
response = generate_response(instruction)
print(response)
```

**Evaluation Checklist:**
- ✅ Test on domain-specific examples
- ✅ Compare with base model
- ✅ Check for overfitting
- ✅ Validate instruction-following capability

---

### **Save LoRA Adapter**

**Objective:** Save only the LoRA adapter weights for efficient storage and deployment.

#### Key Concepts

1. **Adapter Saving** - Save only trainable adapter weights
2. **Storage Efficiency** - Adapters are much smaller than full model
3. **Model Loading** - Load base model + adapters separately
4. **Deployment** - Deploy fine-tuned models efficiently

---

#### Implementation

```python
# Save adapter
model.save_pretrained("./lora_adapter")

# Adapter files saved:
# - adapter_config.json (configuration)
# - adapter_model.bin (weights, ~100MB)

# Load adapter later
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "./lora_adapter")
```

**Storage Comparison:**
- Full fine-tuned model: ~13GB
- LoRA adapter: ~100MB (130x smaller!)

**Benefits:**
- ✅ Easy to share and deploy
- ✅ Multiple adapters for different tasks
- ✅ Version control friendly
- ✅ Quick model switching

---

### **Architecture Overview**

**Complete QLoRA System Architecture:**

```
Base Model (Mistral 7B)
        ↓
4-bit Quantization (BitsAndBytes)
        ↓
QLoRA Training (PEFT)
        ↓
LoRA Adapter (~100MB)
        ↓
Domain-specific chatbot
```

**Why Companies Use This Approach:**

| Method | Cost | GPU Requirements | Training Time |
|--------|------|------------------|--------------|
| Full fine-tuning | $$$$ | Many GPUs (A100) | Days |
| QLoRA | $ | 1 GPU (T4/V100) | Hours |

**That's why most startups use QLoRA + adapters.**

---

### **DPO (Direct Preference Optimization) Alignment**

**Objective:** Align fine-tuned models to human preferences, improving response quality, style, and safety without the complexity of full RLHF.

---

#### Understanding the Problem: Why Alignment Matters

After supervised fine-tuning (SFT), your model can generate correct information but may not always produce responses in the preferred style, tone, or format. For example:

**Fine-Tuned Model Output (Before DPO):**
```
Instruction: "Explain machine learning"

Response: "ML is AI stuff that learns from data. It's like magic but with math."
```

**Desired Output (After DPO):**
```
Instruction: "Explain machine learning"

Response: "Machine learning is a subset of artificial intelligence that enables 
systems to learn from data without explicit programming. It uses algorithms to 
identify patterns and make predictions."
```

**DPO helps bridge this gap** by training the model to prefer high-quality, well-formatted responses over casual or incomplete ones.

---

#### What is RLHF? (The Traditional Approach)

**RLHF (Reinforcement Learning from Human Feedback)** is the traditional method for aligning models to human preferences, used by models like ChatGPT.

**Traditional RLHF Pipeline:**

```
1. Supervised Fine-Tuning (SFT)
   ↓
2. Reward Model Training
   - Collect human preference data (response A vs response B)
   - Train a separate reward model to score outputs
   ↓
3. Reinforcement Learning (PPO)
   - Use reward model to guide training
   - Optimize policy to maximize rewards
   ↓
4. Aligned Model
```

**RLHF Components:**

- **Reward Model**: A separate model trained to predict human preferences
- **Policy Model**: The model being fine-tuned (your LLM)
- **PPO Algorithm**: Reinforcement learning algorithm that optimizes policy
- **Human Feedback**: Preference pairs (response A vs response B)

**Example RLHF Flow:**

```
User Query: "Explain quantum computing"

Model generates Response A: "Quantum computing uses qubits..."
Model generates Response B: "Quantum computing is a type of..."

Human rater: Response A is better (preferred)

Reward Model learns: Response A gets higher score
RL Algorithm: Updates model to generate more responses like A
```

**Challenges with RLHF:**

- ❌ **Complex to implement** - Requires 3 separate models (SFT, reward, policy)
- ❌ **Expensive** - Need to train and maintain a reward model
- ❌ **Unstable training** - RL training (PPO) is notoriously finicky
- ❌ **Hard to debug** - Multiple moving parts make troubleshooting difficult
- ❌ **Computational cost** - Requires significant compute resources

---

#### What is DPO? (The Simpler Alternative)

**DPO (Direct Preference Optimization)** is a simpler alternative to RLHF that directly optimizes model preferences without needing a separate reward model or RL training.

**DPO Pipeline:**

```
1. Supervised Fine-Tuning (SFT)
   ↓
2. Collect Preference Data
   - Same preference pairs as RLHF
   - (response A: preferred, response B: less preferred)
   ↓
3. Direct Optimization
   - Optimize model directly on preferences
   - No reward model needed!
   - No RL needed!
   ↓
4. Aligned Model
```

**Key Innovation:**

DPO uses a mathematical trick to eliminate the need for a reward model:
- Instead of training a reward model, DPO directly optimizes the model on preference pairs
- Uses a special loss function that implicitly learns preferences
- Simpler, faster, and often more stable than RLHF

**DPO vs RLHF Comparison:**

| Aspect | RLHF | DPO |
|--------|------|-----|
| **Steps** | 3 steps (SFT → Reward → RL) | 2 steps (SFT → DPO) |
| **Reward Model** | Required ❌ | Not needed ✅ |
| **RL Algorithm** | Required (PPO) ❌ | Not needed ✅ |
| **Complexity** | High ❌ | Low ✅ |
| **Stability** | Can be unstable ❌ | More stable ✅ |
| **Training Speed** | Slower ❌ | Faster ✅ |
| **Performance** | Baseline | Matches RLHF ✅ |
| **Best For** | Large-scale production | Most use cases ⭐ |

---

#### How DPO Works

**Step 1: Collect Preference Data**

You need preference pairs showing which responses are better:

```python
preference_data = [
    {
        "instruction": "Explain machine learning",
        "response_a": "Machine learning is a subset of artificial intelligence...",  # Preferred
        "response_b": "ML is AI stuff that learns from data...",  # Less preferred
        "preference": "A"  # Human rater preferred A
    },
    # ... more preference pairs
]
```

**Step 2: DPO Training**

DPO trains the model to assign higher probability to preferred responses:

```python
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=reference_model,  # Original SFT model
    args=training_args,
    train_dataset=preference_dataset,
)

dpo_trainer.train()
```

**What Happens During Training:**

- Model learns to prefer response A over response B
- Uses a special loss function that compares probabilities
- Updates only the LoRA adapter weights (base model stays frozen)
- No reward model needed - preferences learned directly

**Step 3: Save Updated Adapter**

After DPO training, save the updated adapter:

```python
# Save adapter with DPO improvements
model.save_pretrained("./lora_adapter_dpo")

# The adapter now contains:
# - Original SFT improvements (from step 1)
# - DPO alignment improvements (from step 2)
```

---

#### Complete Workflow: From Training to Production

**Development/Training Phase:**

```
┌─────────────────────────────────┐
│  STEP 1: Supervised Fine-Tuning │
│  - Train on instruction-response│
│  - Save: lora_adapter_sft/     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  STEP 2: DPO Training           │
│  - Train on preference pairs    │
│  - Updates LoRA adapter weights │
│  - Save: lora_adapter_dpo/      │ ← Contains SFT + DPO
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  STEP 3: Evaluation             │
│  - Test on validation set       │
│  - Compare with base model      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  STEP 4: Model Registry         │
│  - Upload to HuggingFace Hub    │
│  - Version control              │
└──────────────┬──────────────────┘
               │
               ▼
         PRODUCTION
```

**Production Phase:**

```python
# Load pre-trained DPO-aligned model
base_model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, "./lora_adapter_dpo")

# Serve requests (inference only, no training)
response = model.generate(...)
```

**Key Points:**

- ✅ **DPO is a training step** - Done in development, not production
- ✅ **Save adapter after DPO** - Contains both SFT and DPO improvements
- ✅ **Production uses pre-trained model** - Load saved adapter, inference only
- ✅ **No training in production** - Model is read-only

---

#### What Gets Updated During DPO?

**LoRA Adapters:**

- DPO training updates the LoRA adapter weights (matrices A and B)
- Base model stays frozen (quantized, unchanged)
- Same adapter structure, but with improved weights
- Adapter size stays the same (~100MB)

**Model Behavior:**

```
Before DPO:
Base Model (frozen) + LoRA Adapter (SFT-trained)
→ Generates responses based on fine-tuning data
→ May not match preferred style/quality

After DPO:
Base Model (still frozen) + LoRA Adapter (SFT + DPO-trained)
→ Generates responses aligned with human preferences
→ Better quality, style, and safety
```

**Storage:**

- **Before DPO**: LoRA adapter ~100MB (SFT only)
- **After DPO**: LoRA adapter ~100MB (SFT + DPO, same size!)
- DPO doesn't add new parameters, it updates existing adapter weights

---

#### Implementation Example

**Complete DPO Workflow:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# Step 1: Load SFT model
base_model = AutoModelForCausalLM.from_pretrained(...)
sft_model = PeftModel.from_pretrained(base_model, "./lora_adapter_sft")

# Step 2: Prepare preference data
preference_dataset = Dataset.from_list([
    {
        "prompt": "Explain machine learning",
        "chosen": "Machine learning is a subset of AI...",  # Preferred
        "rejected": "ML is AI stuff...",  # Less preferred
    },
    # ... more pairs
])

# Step 3: Configure DPO training
dpo_config = DPOConfig(
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
)

# Step 4: Train with DPO
dpo_trainer = DPOTrainer(
    model=sft_model,
    ref_model=base_model,  # Reference model (original SFT)
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()

# Step 5: Save DPO-aligned adapter
sft_model.save_pretrained("./lora_adapter_dpo")

# Step 6: Deploy to production
production_model = PeftModel.from_pretrained(base_model, "./lora_adapter_dpo")
```

---

#### Benefits of DPO

**Compared to RLHF:**

- ✅ **Simpler** - No reward model, no RL algorithm
- ✅ **Faster** - Direct optimization, fewer steps
- ✅ **More stable** - No RL training instability
- ✅ **Easier to debug** - Single training step
- ✅ **Matches performance** - Achieves similar results to RLHF

**General Benefits:**

- ✅ **Better response quality** - Aligned with human preferences
- ✅ **Improved safety** - Ensures outputs meet safety constraints
- ✅ **Preferred style** - Matches desired tone and format
- ✅ **Consistent outputs** - More reliable responses
- ✅ **Cost-effective** - Lower compute requirements than RLHF

---

#### When to Use DPO

**Use DPO when:**

- ✅ You have preference data (response A vs response B)
- ✅ You want to improve response quality/style
- ✅ You need safety alignment
- ✅ You want simpler alternative to RLHF
- ✅ You have limited compute resources

**Don't use DPO when:**

- ❌ You don't have preference data
- ❌ SFT already meets your requirements
- ❌ You need maximum performance (consider RLHF)
- ❌ You're doing research on alignment methods

---

#### Where to Collect Preference Data: Production vs Development

**This is a critical question:** Should you collect preference pairs (response A/B) in production or development?

**The Answer: Both! Use a hybrid approach.**

---

#### Option 1: Collect in Production (Recommended for Real-World Data)

**Why Collect in Production:**

✅ **Real user queries** - Actual questions users ask
✅ **Real-world preferences** - What users actually prefer
✅ **Production context** - Responses used in real scenarios
✅ **Scale** - Can collect large amounts of data
✅ **Diversity** - Wide variety of use cases

**How to Collect in Production:**

```python
# Production: A/B Testing Approach
def serve_with_preference_collection(user_query):
    # Generate two responses
    response_a = model.generate(user_query, temperature=0.7)
    response_b = model.generate(user_query, temperature=0.9)
    
    # Show both to user (or randomly show one, track which)
    # Collect implicit feedback (which did user engage with?)
    # OR explicit feedback (ask user to rate)
    
    # Store preference pair
    store_preference_pair({
        "query": user_query,
        "response_a": response_a,
        "response_b": response_b,
        "preference": user_feedback,  # A or B
    })
    
    # Return preferred response
    return response_a if user_feedback == "A" else response_b
```

**Production Collection Methods:**

1. **A/B Testing** - Serve two responses, track which performs better
2. **Explicit Rating** - Ask users to rate responses (1-5 stars)
3. **Implicit Feedback** - Track engagement (clicks, time spent, follow-up questions)
4. **Human Evaluation** - Have human raters review production responses

**Challenges:**

- ❌ **Infrastructure needed** - Requires A/B testing system
- ❌ **Privacy concerns** - Need to handle user data carefully
- ❌ **Cost** - Generating multiple responses costs more
- ❌ **User experience** - Showing multiple responses may confuse users

---

#### Option 2: Generate in Development (Recommended for Initial Training)

**Why Generate in Development:**

✅ **Controlled environment** - Easy to generate multiple responses
✅ **No production overhead** - Doesn't affect user experience
✅ **Fast iteration** - Can quickly create preference pairs
✅ **Privacy** - No real user data concerns
✅ **Cost-effective** - Can use cheaper models for generation

**How to Generate in Development:**

```python
# Development: Generate Preference Pairs
def generate_preference_pairs(instructions, model):
    preference_data = []
    
    for instruction in instructions:
        # Generate multiple responses with different parameters
        response_a = model.generate(instruction, temperature=0.7, top_p=0.9)
        response_b = model.generate(instruction, temperature=0.9, top_p=0.95)
        
        # Use LLM-as-judge or human raters to determine preference
        preference = llm_judge(instruction, response_a, response_b)
        
        preference_data.append({
            "instruction": instruction,
            "response_a": response_a,
            "response_b": response_b,
            "preference": preference,
        })
    
    return preference_data
```

**Development Generation Methods:**

1. **LLM-as-Judge** - Use GPT-4 to rate responses
2. **Human Raters** - Internal team rates responses
3. **Rule-based** - Use heuristics (length, format, etc.)
4. **Synthetic Data** - Generate diverse instruction-response pairs

**Advantages:**

- ✅ **Fast** - Can generate thousands of pairs quickly
- ✅ **Controlled** - Can target specific domains or issues
- ✅ **No production impact** - Doesn't affect users
- ✅ **Cost-effective** - Can use cheaper models

---

#### Recommended Hybrid Approach

**Best Practice: Combine Both Methods**

```
┌─────────────────────────────────┐
│  INITIAL TRAINING (Development) │
├─────────────────────────────────┤
│ 1. Generate preference pairs    │
│    - Use LLM-as-judge          │
│    - Human raters               │
│    - Synthetic data             │
│    ↓                            │
│ 2. Train initial DPO model      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  PRODUCTION DEPLOYMENT          │
├─────────────────────────────────┤
│ 3. Deploy DPO-aligned model     │
│    ↓                            │
│ 4. Collect real preference data │ ← Production collection
│    - A/B testing                │
│    - User feedback              │
│    - Implicit signals           │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  CONTINUOUS IMPROVEMENT         │
├─────────────────────────────────┤
│ 5. Retrain with production data │
│    ↓                            │
│ 6. Redeploy improved model     │
└─────────────────────────────────┘
```

**Why This Works:**

1. **Start with Development Data** - Fast initial training
2. **Deploy to Production** - Get real-world performance
3. **Collect Production Data** - Gather actual user preferences
4. **Retrain Periodically** - Improve model with real data
5. **Iterate** - Continuous improvement cycle

---

#### Production Collection Strategies

**Strategy 1: A/B Testing (Recommended)**

```python
# Serve two responses randomly, track which performs better
if random.random() < 0.5:
    response = generate_response_a(query)
    track_response("A", query, response, user_engagement)
else:
    response = generate_response_b(query)
    track_response("B", query, response, user_engagement)

# Analyze which response gets better engagement
preference = analyze_engagement("A", "B")  # "A" or "B"
```

**Strategy 2: Explicit User Rating**

```python
# Show response, ask user to rate
response = model.generate(query)
user_rating = show_rating_ui(response)  # 1-5 stars

# Compare with baseline or other responses
if user_rating >= 4:
    store_preference(response, "preferred")
else:
    store_preference(response, "less_preferred")
```

**Strategy 3: Implicit Feedback**

```python
# Track user behavior
response = model.generate(query)
show_response(response)

# Track engagement signals
signals = {
    "clicked": user_clicked_response,
    "time_spent": time_on_page,
    "follow_up": user_asked_followup,
    "satisfaction": user_satisfaction_score,
}

# Use signals to infer preference
preference = infer_preference_from_signals(signals)
```

---

#### Data Collection Best Practices

**For Initial Training (Development):**

- ✅ Generate 1000-5000 preference pairs
- ✅ Use LLM-as-judge for quick labeling
- ✅ Cover diverse instruction types
- ✅ Include edge cases and failure modes
- ✅ Validate with human raters on sample

**For Production Collection:**

- ✅ Collect continuously (not just once)
- ✅ Use A/B testing for unbiased data
- ✅ Respect user privacy (anonymize data)
- ✅ Collect diverse queries (not just popular ones)
- ✅ Monitor data quality

**Data Quality Guidelines:**

- ✅ **Diverse instructions** - Cover all use cases
- ✅ **Clear preferences** - Obvious which is better
- ✅ **Balanced pairs** - Mix of easy and hard comparisons
- ✅ **Domain coverage** - All relevant domains represented
- ✅ **Quality over quantity** - Better to have 1000 good pairs than 10000 bad ones

---

#### Continuous Learning Workflow

**Complete Cycle:**

```
1. INITIAL TRAINING (Development)
   - Generate preference pairs in dev
   - Train DPO model
   - Deploy to production
   
2. PRODUCTION COLLECTION
   - Collect real user preferences
   - Store preference pairs
   - Monitor data quality
   
3. PERIODIC RETRAINING (Development)
   - Combine dev + production data
   - Retrain DPO model
   - Evaluate improvements
   
4. REDEPLOYMENT
   - Deploy updated model
   - Monitor performance
   - Repeat cycle
```

**Example Implementation:**

```python
# Step 1: Initial training (development)
dev_preferences = generate_preference_pairs(instructions, model)
dpo_trainer.train(dev_preferences)
model.save_pretrained("./lora_adapter_dpo_v1")
deploy_to_production("./lora_adapter_dpo_v1")

# Step 2: Collect in production (over time)
production_preferences = collect_preferences_from_production()

# Step 3: Retrain periodically (development)
combined_preferences = dev_preferences + production_preferences
dpo_trainer.train(combined_preferences)
model.save_pretrained("./lora_adapter_dpo_v2")
deploy_to_production("./lora_adapter_dpo_v2")
```

---

#### Summary: Production vs Development Collection

| Aspect | Development Generation | Production Collection |
|--------|----------------------|---------------------|
| **Speed** | Fast ✅ | Slower (real users) |
| **Cost** | Low ✅ | Higher (A/B testing) |
| **Realism** | Synthetic ❌ | Real-world ✅ |
| **Scale** | Limited | Large scale ✅ |
| **Privacy** | No concerns ✅ | Privacy needed |
| **Best For** | Initial training ✅ | Continuous improvement ✅ |

**Recommendation:**

- **Start with development** - Generate initial preference pairs quickly
- **Deploy to production** - Get real-world model
- **Collect in production** - Gather actual user preferences
- **Retrain periodically** - Combine both data sources
- **Iterate** - Continuous improvement cycle

**Key Insight:** Production data is more valuable but harder to collect. Development data is easier to generate but may not reflect real user needs. The best approach combines both!

---

#### Summary

**DPO (Direct Preference Optimization)** is a simpler, more efficient alternative to RLHF that:

1. **Trains in development** - Not a production process
2. **Updates LoRA adapters** - Modifies adapter weights, keeps base model frozen
3. **Requires preference data** - Needs response pairs (preferred vs less preferred)
4. **Saves updated adapter** - Contains both SFT and DPO improvements
5. **Deploys to production** - Production uses pre-trained DPO-aligned model
6. **Matches RLHF performance** - Achieves similar results with less complexity

**Key Takeaway:** DPO is a training step that happens after SFT to align your model with human preferences. After training, you save the updated adapter and deploy it to production for inference-only use.

---

### **Hybrid RAG + Fine-Tuning System**

**Combining RAG with fine-tuned models for optimal performance:**

```
User Question
     │
     ▼
┌─────────────┐
│ RAG         │ ← Retrieve relevant context from knowledge base
│ Retrieval   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Fine-tuned  │ ← Generate domain-specific answer using
│ LLM         │   retrieved context + fine-tuned knowledge
└──────┬──────┘
       │
       ▼
Better Domain Answer
```

**Why Hybrid?**
- ✅ RAG provides up-to-date information
- ✅ Fine-tuning provides domain expertise
- ✅ Best of both worlds
- ✅ More accurate and contextual responses

---

## 🚀 Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (T4, V100, A100, etc.)
- 8GB+ GPU memory (for 7B models with QLoRA)

### Installation

```bash
# Navigate to project directory
cd tuning_llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft bitsandbytes
pip install datasets
pip install jupyter
```

### Environment Configuration

```bash
# Set CUDA device (if multiple GPUs)
export CUDA_VISIBLE_DEVICES=0

# For Google Colab
# Runtime → Change runtime → GPU
```

---

## 💻 Usage

### Run Jupyter Notebook

```bash
jupyter notebook Fine_tuning_with_PEFT.ipynb
```

### Quick Start

1. **Load Model**: Run cells to load 4-bit quantized model
2. **Add Adapters**: Configure and add LoRA adapters
3. **Prepare Data**: Format your instruction-response dataset
4. **Train**: Run training loop with QLoRA
5. **Test**: Evaluate fine-tuned model
6. **Save**: Save LoRA adapter for deployment

### Example Training Data Format

```python
training_data = [
    {
        "instruction": "Explain quantum computing",
        "response": "Quantum computing uses quantum mechanical phenomena..."
    },
    {
        "instruction": "What is machine learning?",
        "response": "Machine learning is a subset of AI..."
    },
    # ... more examples
]
```

---

## 📂 Project Structure

```
tuning_llm/
├── README.md                    # This file
├── Fine_tuning_with_PEFT.ipynb # Main notebook
└── results/                     # Training outputs (created during training)
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── ...
```

### Key Files

- **`Fine_tuning_with_PEFT.ipynb`**: Complete QLoRA fine-tuning tutorial
  - Model loading with quantization
  - LoRA adapter configuration
  - Training loop
  - Model testing and evaluation
  - Adapter saving
  - DPO alignment overview

---

## 🎓 Learning Objectives

By completing this notebook, you will learn to:

✅ Load and quantize large language models for efficient training

✅ Configure and add LoRA adapters using PEFT

✅ Fine-tune models on custom datasets with QLoRA

✅ Evaluate fine-tuned model performance

✅ Save and deploy LoRA adapters efficiently

✅ Understand DPO alignment for model preferences

✅ Build hybrid RAG + fine-tuning systems

✅ Optimize training for memory-constrained environments

---

## 🔗 Resources

- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)

---

## 🏆 Best Practices

### Model Selection
- **Base Model**: Choose appropriate base model for your task
- **Model Size**: Balance between performance and resource constraints
- **Quantization**: Use 4-bit for maximum memory savings

### Training Configuration
- **Learning Rate**: Start with 2e-4, adjust based on results
- **Batch Size**: Use gradient accumulation for effective larger batches
- **Epochs**: Monitor for overfitting, typically 3-5 epochs
- **Rank (r)**: Lower rank (8-16) for efficiency, higher (32-64) for quality

### Data Preparation
- **Quality over Quantity**: Focus on high-quality examples
- **Domain Coverage**: Ensure diverse examples from your domain
- **Format Consistency**: Use consistent instruction-response format
- **Data Size**: Start with 100-1000 examples, scale as needed

### Memory Optimization
- **Gradient Checkpointing**: Enable for memory savings
- **8-bit Optimizer**: Use `paged_adamw_8bit` for efficiency
- **Batch Size**: Reduce if OOM errors occur
- **Mixed Precision**: Use FP16 for faster training

### Evaluation
- **Test Set**: Reserve 10-20% for evaluation
- **Baseline Comparison**: Compare with base model
- **Domain Metrics**: Use domain-specific evaluation metrics
- **Overfitting Check**: Monitor train vs. validation performance

---

## 🚨 Security Notes

⚠️ **Model Weights**: Be cautious when sharing fine-tuned models. Ensure compliance with model licenses.

⚠️ **Data Privacy**: Protect sensitive data in training datasets. Use data anonymization when necessary.

⚠️ **GPU Resources**: Monitor GPU usage and costs, especially in cloud environments.

⚠️ **Model Deployment**: Validate model outputs before production deployment, especially for safety-critical applications.

---

## 📊 Example Outputs

### Before Fine-Tuning (Base Model)

```
Instruction: "Explain quantum computing in medical applications"

Response: "Quantum computing is a type of computing that uses quantum 
mechanical phenomena. It has potential applications in various fields 
including cryptography and optimization problems."
```

### After Fine-Tuning (Domain-Specific)

```
Instruction: "Explain quantum computing in medical applications"

Response: "Quantum computing holds significant promise for medical 
applications, particularly in drug discovery and molecular simulation. 
Quantum algorithms can model complex molecular interactions more 
efficiently than classical computers, potentially accelerating the 
development of new pharmaceuticals. For example, quantum computers 
could simulate protein folding, enabling researchers to understand 
disease mechanisms and design targeted therapies..."
```

**Improvements:**
- ✅ More domain-specific knowledge
- ✅ Better instruction following
- ✅ More detailed and relevant responses
- ✅ Improved context understanding

---

## ⚖️ PEFT vs. Other Fine-Tuning Methods

### When is PEFT the Best Choice?

**PEFT (especially QLoRA) is optimal when:**

✅ **Resource Constraints** - Limited GPU memory (consumer GPUs, single GPU)
✅ **Cost Efficiency** - Need to minimize training costs (~90% reduction)
✅ **Multiple Tasks** - Want to train multiple adapters for different tasks
✅ **Model Preservation** - Need to keep base model unchanged
✅ **Quick Iteration** - Fast experimentation and hyperparameter tuning
✅ **Startup/Research** - Limited compute budget

**Performance:** PEFT can match or exceed full fine-tuning in many cases (87.2% vs 86.4% on GLUE tasks), but performance varies by domain.

---

### QLoRA vs LoRA: Key Differences

**QLoRA = LoRA + 4-bit Quantization**

| Aspect | LoRA | QLoRA |
|--------|------|-------|
| **Base Model Format** | FP16/BF16 (~13GB for 7B) | 4-bit quantized (~4GB for 7B) |
| **Memory Usage** | ~13GB | ~4GB (3× reduction) |
| **Parameters Trained** | Same (~0.1-1%) | Same (~0.1-1%) |
| **Training Speed** | Same (2× faster than full FT) | Same (2× faster than full FT) |
| **Performance** | Baseline | Matches LoRA (no loss) |
| **GPU Requirements** | T4/V100 (16GB+) | T4/V100 (8GB+) |
| **Best For** | When quantization not needed | **Most use cases** ⭐ |

**Key Insight:** QLoRA is essentially LoRA with quantization added. It provides the same performance as LoRA but uses 3× less memory, making it accessible on smaller GPUs.

**When to use LoRA instead of QLoRA:**
- ✅ Already have sufficient GPU memory (16GB+)
- ✅ Want to avoid any potential quantization artifacts
- ✅ Need maximum precision for research/comparison
- ✅ Working with models that don't quantize well

**When to use QLoRA (recommended):**
- ✅ Limited GPU memory (<16GB)
- ✅ Want maximum efficiency
- ✅ Training on consumer GPUs
- ✅ **Most practical use cases** ⭐

---

### Comparison Table

| Method | Parameters Trained | Memory Usage | Training Speed | Cost | Best For |
|--------|------------------|--------------|----------------|------|----------|
| **Full Fine-Tuning** | 100% (~7B params) | ~40GB+ | Baseline | $$$$ | Maximum performance, large datasets |
| **QLoRA (PEFT)** | 0.1-1% (~10M params) | ~4GB | 2× faster | $ | **Most use cases** ⭐ |
| **LoRA (PEFT)** | 0.1-1% (~10M params) | ~13GB | 2× faster | $$ | Single GPU, no quantization needed |
| **Adapter Layers** | 0.5-5% (~50M params) | ~15GB | 1.5× faster | $$ | Task-specific modules |
| **Prefix Tuning** | <0.1% (~1M params) | ~13GB | 2× faster | $ | Prompt-based tasks |
| **Prompt Tuning** | <0.01% (~100K params) | ~13GB | 2× faster | $ | Few-shot learning |
| **BitFit** | <0.1% (bias only) | ~13GB | 2× faster | $ | Minimal changes needed |

---

### When Full Fine-Tuning is Better

**Full fine-tuning is preferable when:**

✅ **Maximum Performance Required** - Need absolute best results
✅ **Large Dataset** - Millions of examples (>1M samples)
✅ **Complex Tasks** - Coding, mathematics, or highly specialized domains
✅ **Sufficient Resources** - Multiple A100 GPUs available
✅ **Production Critical** - Performance is more important than cost
✅ **Single Task Focus** - Only need one fine-tuned model

**Performance Gap:** Full fine-tuning can outperform PEFT by 5-15% on complex tasks like code generation and mathematical reasoning.

---

### When Other PEFT Methods Beat LoRA/QLoRA

**Adapter Layers** - Better when:
- Need task-specific modules that can be easily swapped
- Want more explicit control over which layers adapt
- Working with transformer architectures that benefit from layer-specific adapters

**Prefix/Prompt Tuning** - Better when:
- Very limited compute (even less than QLoRA)
- Working with prompt-based tasks
- Need to fine-tune without modifying model weights at all
- Want to experiment with different prompt strategies

**BitFit** - Better when:
- Minimal changes needed (only bias terms)
- Extremely resource-constrained environments
- Quick experiments to test if fine-tuning helps

---

### Performance by Domain

| Domain | Full Fine-Tuning | QLoRA | Winner |
|--------|------------------|-------|--------|
| **General NLP** | 86.4% | 87.2% | QLoRA ⭐ |
| **Code Generation** | 85% | 72% | Full Fine-Tuning |
| **Mathematics** | 78% | 65% | Full Fine-Tuning |
| **Domain-Specific** | 82% | 81% | Tie |
| **Instruction Following** | 88% | 87% | Tie |
| **Few-Shot Learning** | 75% | 76% | QLoRA ⭐ |

**Key Insight:** PEFT excels at general NLP and instruction-following, while full fine-tuning is better for complex reasoning tasks.

---

### Decision Framework

**Choose QLoRA if:**
```
IF (GPU Memory < 16GB) OR (Cost is Primary Concern) OR 
   (Multiple Tasks) OR (Quick Iteration Needed):
    → Use QLoRA ✅ (Recommended for most cases)
```

**Choose LoRA if:**
```
IF (GPU Memory >= 16GB) AND (Want to Avoid Quantization) AND
   (Need Maximum Precision):
    → Use LoRA ✅
```

**Choose QLoRA/PEFT if:**
```
IF (GPU Memory < 16GB) OR (Cost is Primary Concern) OR 
   (Multiple Tasks) OR (Quick Iteration Needed):
    → Use QLoRA/PEFT ✅
```

**Choose Full Fine-Tuning if:**
```
IF (GPU Memory > 40GB) AND (Maximum Performance Required) AND 
   (Large Dataset > 1M samples) AND (Complex Domain):
    → Use Full Fine-Tuning ✅
```

**Choose Other PEFT Methods if:**
```
IF (Need Task-Specific Modules):
    → Use Adapter Layers
IF (Extremely Limited Resources):
    → Use Prefix/Prompt Tuning
IF (Minimal Changes Needed):
    → Use BitFit
```

---

### Real-World Recommendations

**For Most Users (90% of cases):**
- **Start with QLoRA** - Best balance of performance, cost, and efficiency
- **Upgrade to full fine-tuning** only if QLoRA doesn't meet performance requirements
- **Use multiple LoRA adapters** for different tasks

**For Enterprises:**
- **Use QLoRA for experimentation** and rapid prototyping
- **Use full fine-tuning for production** models where performance is critical
- **Hybrid approach**: QLoRA for most tasks, full fine-tuning for flagship products

**For Researchers:**
- **QLoRA for most experiments** - Faster iteration, lower costs
- **Full fine-tuning for final models** - Maximum performance for publications
- **Compare both methods** - Report results from both approaches

---

## 🎯 Next Steps

After mastering QLoRA fine-tuning, consider:

1. **Advanced Techniques**:
   - Multi-task fine-tuning with multiple adapters
   - Adapter composition and merging
   - Continual learning with adapters

2. **Optimization**:
   - Hyperparameter tuning (rank, alpha, learning rate)
   - Experiment with different base models
   - Optimize training data quality

3. **Deployment**:
   - Model serving with vLLM or TGI
   - API endpoints for fine-tuned models
   - Integration with RAG systems

4. **Alignment**:
   - Implement DPO training
   - Collect preference data
   - Improve response quality and safety

5. **Scaling**:
   - Fine-tune larger models (13B, 70B)
   - Distributed training strategies
   - Production-grade deployment

---

## 📄 License

This project is for educational purposes.

---

**Happy Fine-Tuning! 🚀**
