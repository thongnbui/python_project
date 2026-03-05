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

1. **4-bit Quantization** - Reduce model precision from FP16 to 4-bit integers
2. **BitsAndBytes Integration** - Use BitsAndBytes library for quantization
3. **Model Loading** - Load base model with quantization config
4. **Memory Optimization** - Enable efficient memory usage for training

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

#### Key Concepts

1. **LoRA (Low-Rank Adaptation)** - Add small trainable matrices to attention layers
2. **Parameter Efficiency** - Only train ~0.1-1% of model parameters
3. **Adapter Configuration** - Configure rank, alpha, and target modules
4. **PEFT Integration** - Use PEFT library for LoRA implementation

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

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Add LoRA adapters
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
```

**Output Example:**
```
trainable params: 8,388,608 || all params: 3,411,550,208 || trainable%: 0.25
```

**Key Benefits:**
- ✅ Only ~0.25% of parameters are trainable
- ✅ Adapters can be saved/loaded independently
- ✅ Multiple adapters can be swapped for different tasks
- ✅ Base model remains unchanged

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

**Objective:** Align fine-tuned models to human preferences without full RLHF.

#### Key Concepts

1. **Preference Learning** - Learn from human preference pairs
2. **Alignment** - Improve response quality and style
3. **Efficiency** - Simpler than full RLHF pipeline
4. **Safety** - Ensure model outputs meet safety constraints

---

#### What is DPO?

**After fine-tuning a model on your dataset, DPO helps align the model to preferred outputs (like human preferences) without full RLHF.**

**Why it matters:**
- Fine-tuned LoRA adapters may generate correct info but not always in the preferred style
- DPO "nudges" the model to produce better responses
- Ensures safety and alignment constraints

**How to implement:**

1. **Collect preference pairs**: (response A, response B) for the same instruction
   - Response A: Preferred response
   - Response B: Less preferred response

2. **Train the model** on preference pairs using DPO loss

3. **Evaluate** improved alignment and response quality

**Example Preference Pair:**
```
Instruction: "Explain machine learning"

Response A (Preferred):
"Machine learning is a subset of artificial intelligence that enables 
systems to learn from data without explicit programming. It uses 
algorithms to identify patterns and make predictions."

Response B (Less Preferred):
"ML is AI stuff that learns from data. It's like magic but with math."
```

**Benefits:**
- ✅ Better response quality
- ✅ Improved safety alignment
- ✅ Preferred style and tone
- ✅ No need for full RLHF pipeline

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
