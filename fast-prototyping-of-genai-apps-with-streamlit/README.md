# Fast Prototyping of GenAI Apps with Streamlit

Welcome to the [*Fast Prototyping of GenAI Apps with Streamlit*](https://www.deeplearning.ai/courses/fast-prototyping-of-genai-apps-with-streamlit/) GitHub repository! This comprehensive course teaches you how to rapidly build and deploy GenAI applications using Streamlit and Snowflake, working with the Avalanche dataset - a hypothetical winter sports gear company's customer reviews and shipping logs.

## Course Overview

This course is structured into three progressive modules that take you from Streamlit basics to advanced RAG implementations:

### Module 1: Streamlit Fundamentals & Data Processing
**Focus**: Building interactive data applications with Streamlit and GenAI integration

- **Lesson 1**: Getting started with Streamlit basics
  - Building your first Streamlit app
  - Understanding Streamlit's reactive model
  
- **Lesson 2**: Integrating OpenAI API
  - Setting up OpenAI client and API keys
  - Creating GenAI-powered applications
  - Managing temperature and response parameters
  
- **Lesson 3**: Data Processing & Visualization
  - Data ingestion and cleaning (text processing, parsing)
  - Interactive filtering with dropdowns and multiselect
  - Visualization with multiple libraries (Altair, Matplotlib, Plotly, Streamlit native)
  - Session state management
  - Deploying Streamlit apps

### Module 2: Snowflake Integration & Analytics
**Focus**: Leveraging Snowflake's cloud data platform for scalable GenAI applications

- **Lesson 1**: Snowpark & Data Engineering
  - Working with Snowflake Snowpark DataFrames
  - Loading and staging data in Snowflake
  - Merging datasets (customer reviews + shipping logs)
  - Data cleaning and transformation with Snowpark functions
  
- **Lesson 2**: Snowflake Cortex for GenAI
  - Sentiment analysis using Snowflake Cortex
  - Building interactive dashboards with Snowflake data
  - Creating Q&A chatbots with LLMs (Claude integration)
  - Advanced visualizations and data filtering

### Module 3: Advanced Features - RAG & Search
**Focus**: Building Retrieval-Augmented Generation systems with Snowflake Cortex

- **Lesson 1**: Deployment strategies
  - Production deployment best practices
  
- **Lesson 3**: RAG Implementation
  - Text chunking with `SPLIT_TEXT_RECURSIVE_CHARACTER`
  - Creating Cortex Search Services
  - Embedding with Snowflake Arctic models
  - Querying with both SQL and Python
  - Building context-aware chatbots

## What You'll Build

Throughout this course, you'll build a complete sentiment analysis and customer insight platform:

1. **Data Processing Pipeline**: Ingest, clean, and transform customer reviews
2. **Interactive Dashboards**: Visualize sentiment scores, product ratings, and shipping metrics
3. **GenAI Chatbots**: Answer questions about customer data using LLMs
4. **RAG System**: Search and retrieve relevant customer reviews using semantic search
5. **Production-Ready Apps**: Deploy your applications to the cloud

## Key Technologies

- **Streamlit**: Rapid UI development for data apps
- **OpenAI API**: GPT-4o integration for text generation
- **Snowflake**: Cloud data platform and warehouse
- **Snowflake Cortex**: Built-in AI/ML functions (sentiment analysis, LLMs, search)
- **Snowpark**: DataFrame API for data processing
- **Claude 3.5 Sonnet**: Advanced LLM for conversational AI

## Dataset: Avalanche Winter Sports Gear

The course uses customer data from Avalanche, a hypothetical winter sports company:
- **Customer Reviews**: Product feedback and ratings (structured CSV and unstructured DOCX files)
- **Shipping Logs**: Delivery tracking, carrier info, regional data, late deliveries
- **Combined Analytics**: Sentiment scores, product insights, regional performance

## Repository Structure

Upon opening this repository, you will find:

- **`data`**: Contains the Avalanche dataset needed for the course.

- **Modules (M1, M2 and M3)**: There is a dedicated folder for each module of the course. Within these folders:
  - Code is organized by lessons.
  - Each lesson contains Python files or notebooks as used in the videos.
  - A `lab` folder is available with all necessary resources for lab activities.

- **`requirements.txt`**: Present where necessary to help you install all required dependencies for specific modules.

- Additional files:
  - **`.env.example`**: A sample environment file. Duplicate and rename to `.env`, then add your OpenAI API key.
  - **`README.md`**: This guide to help you get started.

## Getting Started

To work with the course files, it's recommended to clone this repository to your local machine. This allows you to modify code as you proceed through the videos.

### Instructions to Clone Repository

1. Sign in to your GitHub account.
2. Navigate to the main repo page.
3. Look for the green "Code" button near the top right.
4. Copy the repository address (it should start with "git...").
5. Using the command line, type: `git clone [repository address]` if you have git installed.
6. Alternatively, use GitHub Desktop by selecting File > Clone Repository.
7. After cloning, ensure your local copy of the repo is public to facilitate deployment on Snowflake later.

## Utilizing the Course Files

This GitHub repository provides a straightforward project structure, pre-loaded with the necessary datasets and starter files. Each video corresponds to a file set under a naming convention like `M1L1V1` (Module 1, Lesson 1, Video 1) for easy orientation.

You can either follow along with the provided files or write your own code as you progress. The video series typically builds upon previous sessions, allowing you to continue developing your solutions or refer to the provided working solutions within each lesson.

## Learning Path

**Recommended progression**:
1. Start with Module 1 to learn Streamlit basics and OpenAI integration
2. Progress to Module 2 to integrate Snowflake's powerful data platform
3. Advance to Module 3 for RAG and semantic search implementations

Each module includes:
- **Code files**: Step-by-step implementations (some with `_starting.py` templates)
- **Lab folders**: Hands-on exercises with solutions
- **Deploy folders**: Production-ready deployment examples

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GenAI features)
- Snowflake account (for Module 2 and 3)
- Basic understanding of Python and data structures

## Quick Start

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your API keys
4. Navigate to a lesson folder and run: `streamlit run <filename>.py`

Happy coding and building amazing GenAI applications! 🚀
