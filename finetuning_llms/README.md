# Fine-tuning Large Language Models (LLMs)

A comprehensive tutorial series on fine-tuning large language models, covering everything from understanding the basics to implementing, training, and evaluating custom models.

## Overview

This project contains a complete walkthrough of the LLM fine-tuning process, demonstrating how to adapt pre-trained models for specific tasks and domains. The tutorials use practical examples and hands-on code to illustrate concepts and best practices in model customization.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Lesson Overview](#lesson-overview)
- [Key Concepts](#key-concepts)
- [Getting Started](#getting-started)
- [Technologies Used](#technologies-used)
- [Use Cases](#use-cases)
- [Best Practices](#best-practices)

## Prerequisites

- Python 3.9+
- Basic understanding of machine learning concepts
- Familiarity with transformers and LLMs
- Access to GPU resources (recommended for training)
- API keys for Lamini platform (for cloud training)

### Required Packages

```bash
pip install transformers datasets torch pandas jsonlines lamini
```

## Project Structure

```
finetuning_llms/
├── 01_Why_finetuning.ipynb              # Compare finetuned vs. non-finetuned models
├── 02_Where_finetuning_fits_in.ipynb    # Pretraining vs. finetuning data
├── 03_Instruction_tuning.ipynb          # Instruction-based fine-tuning
├── 04_Data_preparation.ipynb            # Tokenization and data formatting
├── 05_Training.ipynb                    # Model training pipeline
├── 06_Evaluation.ipynb                  # Model evaluation and testing
└── README.md                            # This file
```

## Lesson Overview

### 01. Why Fine-tuning?

**Objective:** Understand the difference between base models and fine-tuned models through practical comparisons.

**Topics Covered:**
- Comparing non-finetuned models (Llama-2-7b-hf) with fine-tuned models (Llama-2-7b-chat-hf)
- Real-world examples demonstrating improved performance
- Understanding when fine-tuning is necessary
- Comparison with ChatGPT responses

**Key Insights:**
- Base models often generate repetitive or unfocused outputs
- Fine-tuned models provide more coherent, task-specific responses
- Instruction formatting (e.g., `[INST]...[/INST]`) matters for fine-tuned models

### 02. Where Fine-tuning Fits In

**Objective:** Understand the distinction between pretraining and fine-tuning datasets.

**Topics Covered:**
- Pretraining datasets (Common Crawl, The Pile)
- Fine-tuning datasets (lamini_docs.jsonl)
- Data format differences
- Various data formatting approaches

**Key Concepts:**
- **Pretraining Data:** Large-scale, diverse, unstructured text from the web
- **Fine-tuning Data:** Curated, structured datasets with specific formats (Q&A, instruction-response)
- Flexible formatting: question/answer, instruction/response, input/output, or plain text

**Data Format Examples:**
```python
# Q&A format
{"question": "How do I...", "answer": "You can..."}

# Instruction-Response format
{"instruction": "Explain...", "response": "The explanation is..."}

# Input-Output format
{"input": "Task description", "output": "Expected result"}
```

### 03. Instruction Tuning

**Objective:** Learn how to structure and prepare instruction-based training data.

**Topics Covered:**
- Loading instruction-tuned datasets (Alpaca dataset)
- Prompt template design
- Hydrating prompts with data
- Handling instructions with and without additional input

**Prompt Templates:**
```python
# Template with input context
"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

# Template without input
"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""
```

**Key Skills:**
- Structuring prompts for optimal model performance
- Conditional formatting based on data availability
- Processing datasets for instruction tuning

### 04. Data Preparation

**Objective:** Master tokenization, padding, and data preprocessing techniques.

**Topics Covered:**
- Text tokenization with AutoTokenizer
- Encoding and decoding text
- Batch tokenization
- Padding strategies for variable-length sequences
- Truncation for length constraints
- Preparing data for model training

**Key Operations:**
```python
# Tokenization
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
encoded_text = tokenizer("Hi, how are you?")["input_ids"]

# Padding
tokenizer.pad_token = tokenizer.eos_token
encoded_texts = tokenizer(list_texts, padding=True)

# Truncation
encoded_texts = tokenizer(list_texts, max_length=3, truncation=True)
```

**Best Practices:**
- Set appropriate padding tokens
- Balance between padding overhead and batch efficiency
- Consider maximum sequence length for your use case
- Handle special tokens correctly

### 05. Training

**Objective:** Implement the complete training pipeline for fine-tuning LLMs.

**Topics Covered:**
- Selecting base models (EleutherAI/pythia series)
- Loading training data from JSONL files
- Configuring training parameters
- Training on cloud GPUs with Lamini
- Understanding the training loop
- Using Hugging Face Transformers and datasets

**Simple Training API:**
```python
from llama import BasicModelRunner

model = BasicModelRunner("EleutherAI/pythia-410m") 
model.load_data_from_jsonlines(
    "lamini_docs.jsonl", 
    input_key="question", 
    output_key="answer"
)
model.train(is_public=True)
```

**Key Components:**
1. **Base Model Selection:** Choose appropriate model size for your task
2. **Data Loading:** Format data with proper input/output keys
3. **Training Configuration:** Set hyperparameters and training arguments
4. **Execution:** Run training on GPU infrastructure

### 06. Evaluation

**Objective:** Evaluate fine-tuned model performance and compare with baselines.

**Topics Covered:**
- Loading fine-tuned models
- Creating evaluation datasets
- Implementing evaluation metrics
- Batch inference for efficiency
- Comparing model outputs
- Exact match and similarity metrics

**Evaluation Pipeline:**
```python
# Load test dataset
dataset = datasets.load_dataset("lamini/lamini_docs")
test_dataset = dataset["test"]

# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")
tokenizer = AutoTokenizer.from_pretrained("lamini/lamini_docs_finetuned")

# Evaluate
def is_exact_match(prediction, ground_truth):
    return prediction.strip() == ground_truth.strip()
```

**Evaluation Metrics:**
- Exact match accuracy
- Text similarity (difflib)
- Task-specific metrics
- Qualitative assessment

## Key Concepts

### Fine-tuning vs. Pretraining

| Aspect | Pretraining | Fine-tuning |
|--------|-------------|-------------|
| **Data** | Large-scale web text (billions of tokens) | Curated task-specific data (thousands to millions) |
| **Goal** | Learn general language patterns | Adapt to specific tasks/domains |
| **Compute** | Extremely high (weeks/months on large clusters) | Moderate (hours/days on single GPU) |
| **Output** | Base model | Specialized model |

### When to Fine-tune

✅ **Fine-tune when:**
- You need domain-specific knowledge (medical, legal, technical)
- You want consistent output formatting
- You need to follow specific instructions
- You have quality training data for your use case
- You want to reduce prompt engineering complexity

❌ **Don't fine-tune when:**
- Few-shot prompting works well enough
- You lack quality training data
- Your task changes frequently
- You need general-purpose capabilities

### Model Selection Guide

| Model Size | Parameters | Use Case | Hardware |
|------------|-----------|----------|----------|
| Pythia-70m | 70M | Quick experiments, testing | CPU/Small GPU |
| Pythia-410m | 410M | Moderate tasks, prototypes | Single GPU |
| Pythia-1B+ | 1B+ | Production tasks | Multi-GPU |
| Llama-7B+ | 7B+ | Complex reasoning | High-end GPU |

## Technologies Used

### Core Libraries

- **Transformers (Hugging Face):** Model loading, tokenization, and training
- **Datasets (Hugging Face):** Data loading and processing
- **PyTorch:** Deep learning framework
- **Lamini:** Simplified fine-tuning platform with cloud GPU access

### Models

- **EleutherAI/Pythia:** Open-source models (70M to 12B parameters)
- **Meta/Llama-2:** State-of-the-art base and chat models
- **Custom Fine-tuned Models:** Domain-specific adaptations

### Data Sources

- **Common Crawl (C4):** Pretraining dataset
- **The Pile:** Large-scale diverse dataset
- **Alpaca:** Instruction-following dataset
- **Lamini Docs:** Custom technical documentation dataset

## Use Cases

### 1. Customer Support Automation
Fine-tune on your support ticket history to generate accurate, context-aware responses.

```python
# Example data format
{
    "question": "I didn't receive my item",
    "answer": "I apologize for the inconvenience. Let me help you track your order..."
}
```

### 2. Technical Documentation Generation
Create models that understand your codebase and generate accurate documentation.

```python
# Example data format
{
    "question": "Can Lamini generate technical documentation?",
    "answer": "Yes, Lamini can generate technical documentation using NLG techniques..."
}
```

### 3. Domain-Specific Q&A
Build specialized assistants for medical, legal, or scientific domains.

### 4. Code Generation
Train models on your coding patterns and internal APIs for better code suggestions.

### 5. Creative Writing
Fine-tune for consistent character voices, writing styles, or genre-specific content.

## Best Practices

### Data Quality

1. **Curate High-Quality Examples**
   - Ensure factual accuracy
   - Maintain consistent formatting
   - Include diverse examples
   - Remove noisy or irrelevant data

2. **Balance Your Dataset**
   - Represent different task types
   - Avoid data imbalance
   - Include edge cases
   - Cover failure modes

3. **Data Format Consistency**
   - Use consistent prompt templates
   - Standardize output formatting
   - Document your data schema
   - Validate data before training

### Training

1. **Start Small**
   - Begin with smaller models (70M-410M)
   - Iterate quickly
   - Scale up once you validate the approach

2. **Monitor Training**
   - Track loss curves
   - Validate on held-out data
   - Watch for overfitting
   - Save checkpoints regularly

3. **Hyperparameter Tuning**
   - Learning rate: Start with 1e-5 to 5e-5
   - Batch size: Maximize based on GPU memory
   - Epochs: Usually 3-5 for fine-tuning
   - Warmup steps: 10% of total steps

### Evaluation

1. **Multiple Metrics**
   - Quantitative: accuracy, F1, BLEU
   - Qualitative: human evaluation
   - Task-specific: domain metrics

2. **Test Set Quality**
   - Representative of real use cases
   - Never seen during training
   - Diverse and challenging

3. **Continuous Monitoring**
   - Track performance over time
   - A/B test with baseline
   - Collect user feedback

### Production Considerations

1. **Model Management**
   - Version your models
   - Document training process
   - Track data provenance
   - Maintain model cards

2. **Deployment**
   - Optimize inference speed
   - Implement caching strategies
   - Monitor API usage and costs
   - Set up fallback mechanisms

3. **Safety**
   - Test for biases
   - Implement content filtering
   - Monitor for harmful outputs
   - Establish usage guidelines

## Environment Setup

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install transformers datasets torch pandas jsonlines

# Set up Jupyter
pip install jupyter
jupyter notebook
```

### Cloud Training with Lamini

```bash
# Install Lamini
pip install lamini

# Set environment variables
export POWERML__PRODUCTION__URL="your_api_url"
export POWERML__PRODUCTION__KEY="your_api_key"
```

### GPU Requirements

- **Minimum:** 8GB GPU (for 70M-410M models)
- **Recommended:** 16GB+ GPU (for 1B+ models)
- **Training:** A100 or V100 recommended for larger models

## Troubleshooting

### Common Issues

**Problem:** Out of memory errors
- **Solution:** Reduce batch size, use gradient accumulation, or try a smaller model

**Problem:** Poor model performance
- **Solution:** Check data quality, increase training examples, adjust learning rate

**Problem:** Model not following instructions
- **Solution:** Verify prompt template formatting, ensure consistent data structure

**Problem:** Training too slow
- **Solution:** Use cloud GPUs, enable mixed precision training, optimize data loading

### Getting Help

- Check Hugging Face documentation: https://huggingface.co/docs
- Lamini documentation: https://lamini.ai/docs
- Transformers issues: https://github.com/huggingface/transformers/issues

## Next Steps

1. **Experiment with Different Models:** Try various base models to find the best fit
2. **Collect More Data:** Expand your training dataset with diverse examples
3. **Advanced Techniques:** Explore LoRA, QLoRA, and parameter-efficient fine-tuning
4. **Deploy Your Model:** Set up inference endpoints and integrate with applications
5. **Monitor and Iterate:** Continuously improve based on real-world performance

## Additional Resources

### Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Datasets Library](https://huggingface.co/docs/datasets)
- [PyTorch](https://pytorch.org/docs)
- [Lamini Platform](https://lamini.ai/docs)

### Research Papers
- "Scaling Instruction-Finetuned Language Models" (Flan-T5)
- "Training language models to follow instructions with human feedback" (InstructGPT)
- "Self-Instruct: Aligning Language Models with Self-Generated Instructions"

### Community
- Hugging Face Forums
- ML/AI Subreddits
- Discord communities for LLM developers

## License

This project follows the licenses of the underlying models and datasets used:
- EleutherAI/Pythia: Apache 2.0
- Meta/Llama-2: Llama 2 Community License
- Check individual dataset licenses before use

## Acknowledgments

- Hugging Face for Transformers and Datasets libraries
- EleutherAI for open-source Pythia models
- Lamini for simplified fine-tuning infrastructure
- The open-source ML community for datasets and tools

---

**Note:** This is a learning resource. For production deployments, ensure proper testing, safety measures, and compliance with model licenses and regulations.

