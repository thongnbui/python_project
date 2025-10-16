# Generative AI with Snowflake

## Course Overview
This repository contains all code and materials from the **Introduction to Generative AI with Snowflake** course available on Coursera and LinkedIn Learning. The course provides comprehensive hands-on training on building generative AI applications using Snowflake Cortex AI functions, including LLM integration, fine-tuning, and production deployment.

## 📚 Course Modules

### Module 1: Introduction to Snowflake Cortex
**Topic:** Call Transcript Analysis with LLMs

Learn the fundamentals of Snowflake Cortex by building a call transcript analysis system:
- Setting up Snowflake environment (database, schema, warehouse)
- Loading data from S3 into Snowflake tables
- Using `SNOWFLAKE.CORTEX.COMPLETE()` for text generation
- Comparing different LLMs (Llama, Mistral) for summarization tasks
- Extracting structured JSON from unstructured call transcripts
- Building interactive Streamlit applications within Snowflake

**Key Files:**
- `module-1/call_transcript_analysis.ipynb` - Complete analysis workflow
- `module-1/environment.yml` - Environment configuration

### Module 2: Cortex AI Functions Deep Dive
**Topics:** LLM Functions & Task-Specific Functions

#### Introduction to LLM Functions (`intro_to_LLM_functions.ipynb`)
- **Complete Function**: Text generation with various LLMs
- **Task-Specific Functions**:
  - `TRANSLATE()` - Multi-language translation with auto-detect
  - `SENTIMENT()` - Sentiment analysis scoring (-1 to 1 scale)
  - `SUMMARIZE()` - Text summarization
  - `CLASSIFY_TEXT()` - Text classification with custom categories
- **Helper Functions**:
  - `COUNT_TOKENS()` - Token counting for cost estimation
  - `TRY_COMPLETE()` - Safe completion with error handling

#### Advanced LLM Usage (`using_LLM_functions.ipynb`)
- **System & User Roles**: Setting context and personas for LLMs
- **Cortex Guard**: Built-in content safety and guardrails
- **Multi-turn Conversations**: Chat history and context management
- **Parameter Tuning**:
  - `temperature` - Control randomness/creativity
  - `top_p` - Nucleus sampling for output diversity
  - `max_tokens` - Limit response length
- **Snowpark Integration**: Using Cortex functions in DataFrame operations
- **Few-Shot Classification**: Custom classification with examples

#### Streamlit Application (`call_transcripts_analytics_app.py`)
Production-ready app featuring:
- JSON summary generation from call transcripts
- Multi-language translation (11+ languages)
- Real-time sentiment analysis
- Interactive UI with Snowflake integration

### Module 3: Fine-Tuning LLMs
**Topic:** Custom Model Training for Support Ticket Response Generation

Complete workflow for fine-tuning LLMs on domain-specific tasks:

#### Data Preparation (`load_support_tickets.ipynb`)
- Loading support ticket data from S3
- Creating training datasets for telecommunications support

#### Fine-Tuning Workflow (`finetuning_mistral_7b.ipynb`)
1. **Data Generation & Filtering**:
   - Using Mistral-Large to generate high-quality training data
   - Filtering responses by word count and contact preference
   - Creating prompt-completion pairs

2. **Model Training**:
   - Splitting data into train/eval sets (80/20)
   - Fine-tuning Mistral-7B using `SNOWFLAKE.CORTEX.FINETUNE()`
   - Monitoring training progress and status

3. **Inference & Deployment**:
   - Using fine-tuned models for custom response generation
   - Comparing base model vs fine-tuned performance
   - Building production applications with custom models

#### Support Ticket Response App (`support_ticket_response_app.py`)
Interactive application for automated customer support:
- Auto-categorize support tickets using `CLASSIFY_TEXT()`
- Generate personalized email or SMS responses
- Support for multiple LLMs (base and fine-tuned)
- Word count control for different communication channels

**Ticket Categories Supported:**
- Roaming fees
- Slow data speed
- Lost phone
- Add new line
- Closing account

## 🔬 Additional Demos

### Medical Notes Extraction (`additional-demos/medical_notes_extraction.ipynb`)
Demonstrates structured data extraction from unstructured medical records:
- Extracting patient information, conditions, and interventions
- Comparing different LLM sizes (Llama 3.2-1B, 3.2-3B, 3.1-405B)
- Building form-based extraction systems
- Healthcare-specific NLP applications

## 🛠 Key Technologies

- **Snowflake Cortex AI**: LLM functions and fine-tuning capabilities
- **Snowpark Python**: DataFrame operations with AI functions
- **Streamlit**: Interactive application development
- **LLM Models**:
  - Llama 3.1 (8B, 70B, 405B)
  - Llama 3.2 (1B, 3B)
  - Mistral (7B, Large)
  - Custom fine-tuned models

## 🚀 Applications Built

1. **Call Transcript Analyzer**: Multi-functional app for transcript analysis, translation, and sentiment scoring
2. **Support Ticket Response Generator**: Automated customer support with fine-tuned LLMs
3. **Medical Data Extractor**: Healthcare information extraction system

## 📝 Important Notes

### Cortex AISQL Migration (June 2025)
LLM Functions are now called **Cortex AISQL**. Your existing code using `SNOWFLAKE.CORTEX.*` functions continues to work. New functions with the `AI_` prefix (e.g., `AI_COMPLETE`, `AI_CLASSIFY`) offer additional capabilities and are available in both SQL and Python via Snowpark.

**Resources:**
- [Cortex AISQL Documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql)
- [Using Cortex AISQL with Python](https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql#using-snowflake-cortex-aisql-with-python)

## 🎯 Learning Outcomes

After completing this course, you will be able to:
- Build end-to-end generative AI applications on Snowflake
- Use Cortex AI functions for text generation, translation, sentiment analysis, and classification
- Fine-tune LLMs on custom datasets for domain-specific tasks
- Deploy production-ready Streamlit applications with Snowflake
- Implement multi-turn conversations and context management
- Extract structured data from unstructured text
- Optimize LLM parameters for different use cases
- Handle multilingual data and translation workflows	
