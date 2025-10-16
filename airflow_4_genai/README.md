# Orchestrating Workflows for GenAI Applications with Apache Airflow

## 📚 Overview

This directory contains coursework from the DeepLearning.AI course: **"Orchestrating Workflows for GenAI Applications"**. The course demonstrates how to build and automate a RAG (Retrieval-Augmented Generation) system using Apache Airflow, vector databases, and embeddings.

**Course Link**: [Orchestrating Workflows for GenAI Applications](https://learn.deeplearning.ai/courses/orchestrating-workflows-for-genai-applications/lesson/mdd7p/your-rag-prototype)

---

## 📁 Directory Structure

```
airflow_4_genai/
├── L2/                    # Lesson 2: RAG Prototype
│   ├── L2.ipynb          # Main notebook
│   ├── helper.py         # Utility functions
│   ├── requirements.txt  # Python dependencies
│   ├── include/data/     # Book description files
│   └── tmp/weaviate/     # Local Weaviate database
├── L3/                    # Lesson 3: Building Simple Pipelines
│   ├── L3.ipynb          # Airflow DAG basics
│   ├── helper.py         # Utility functions
│   └── airflow_architecture_3.png
├── L4/                    # (Content in DAG files)
├── L5/                    # Lesson 5: Scheduling & Parameters
│   └── L5.ipynb          # Time-based & data-aware scheduling
└── README.md             # This file
```

---

## 🎯 Learning Objectives

- Build a RAG prototype with vector databases
- Orchestrate GenAI workflows with Apache Airflow
- Implement time-based and data-aware scheduling
- Use Airflow TaskFlow API and DAG patterns
- Create production-ready AI pipelines

---

## 🛠️ Tech Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Orchestration** | Apache Airflow | Latest | Workflow automation |
| **Vector DB** | Weaviate | 4.14.1 | Store embeddings |
| **Embeddings** | FastEmbed | 0.6.1 | Generate vectors (BAAI/bge-small-en-v1.5) |
| **Language** | Python | 3.12+ | Development |
| **Scheduling** | Airflow Assets | - | Data-aware triggers |

---

## 📝 Lesson Breakdown

### **Lesson 2: Your RAG Prototype** 🔬

**Key Concepts**:
- Building a RAG system from scratch
- Creating vector embeddings from text
- Using embedded Weaviate for local vector storage
- Semantic search for book recommendations

**What it does**:
1. Reads book descriptions from text files (`include/data/*.txt`)
2. Creates vector embeddings using FastEmbed (BAAI/bge-small-en-v1.5)
3. Stores embeddings in local Weaviate instance
4. Performs semantic search queries

**Example Query**:
```python
query = "A philosophical book"
# Returns: The Idea of the World (2019) by Bernardo Kastrup
```

**Data Flow**:
```
Text Files → Extract Metadata → Create Embeddings → Weaviate DB → Query Results
```

---

### **Lesson 3: Building a Simple Pipeline** 🔧

**Key Concepts**:
- Airflow architecture and components
- Creating DAGs with Python decorators (`@dag`, `@task`)
- Task dependencies with `chain()` function
- Using the Airflow UI

**Airflow Components**:
1. **DAG Processor** - Parses and serializes DAGs
2. **Scheduler** - Determines task execution
3. **Workers** - Execute tasks
4. **Metadata Database** - Stores state
5. **API Server/UI** - Web interface

**Sample DAGs Created**:
- `my_first_dag.py` - Basic task dependencies
- `my_second_dag.py` - Math operations with parallel tasks

**Example DAG**:
```python
from airflow.sdk import dag, task, chain

@dag
def my_first_dag():
    @task
    def my_task_1():
        return {"my_word": "Airflow!"}
    
    @task
    def my_task_2(my_dict):
        print(my_dict["my_word"])
    
    _task_1 = my_task_1()
    _task_2 = my_task_2(my_dict=_task_1)
```

---

### **Lesson 5: Scheduling and DAG Parameters** ⏰

**Key Concepts**:
- Time-based scheduling (`@hourly`, cron expressions)
- Data-aware scheduling with `Asset` objects
- DAG parameters for dynamic queries
- Converting prototypes to production pipelines

**Two Main DAGs**:

#### 1. **`fetch_data` DAG** (Producer)
- **Schedule**: Runs every hour (`@hourly`)
- **Steps**:
  1. Creates Weaviate collection if not exists
  2. Lists book description files
  3. Transforms text to structured data
  4. Creates vector embeddings
  5. Loads embeddings to Weaviate
  6. **Emits Asset Event** when complete

#### 2. **`query_data` DAG** (Consumer)
- **Schedule**: Triggered by Asset updates (data-aware)
- **Steps**:
  1. Waits for `my_book_vector_data` Asset event
  2. Queries vector DB with custom parameters
  3. Returns book recommendations

**Scheduling Examples**:
```python
# Time-based scheduling
@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly"
)
def fetch_data():
    ...

# Data-aware scheduling
@task(outlets=[Asset("my_book_vector_data")])
def load_embeddings_to_vector_db(...):
    # Emits event when done
    ...

@dag(schedule=[Asset("my_book_vector_data")])
def query_data():
    # Triggers when Asset updates
    ...

# Custom parameters
@dag(params={"query_str": "A philosophical book"})
def query_data():
    ...
```

---

## 🔄 Complete RAG Workflow

```
┌─────────────────┐
│ Book Text Files │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  fetch_data DAG     │ (Hourly)
│  • Extract text     │
│  • Create embeddings│
│  • Store in Weaviate│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Asset Event Emitted │
│ "my_book_vector_data"│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  query_data DAG     │ (Triggered)
│  • Semantic search  │
│  • Return results   │
└─────────────────────┘
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.12+
- Apache Airflow environment (or Astro CLI)
- Docker (for standalone Airflow)

### Install Dependencies
```bash
cd L2
pip install -r requirements.txt
```

**Dependencies**:
```txt
weaviate-client==4.14.1  # Vector database
fastembed==0.6.1         # Embedding generation
ipython                  # Interactive notebooks
```

### Run Notebooks
```bash
jupyter notebook L2/L2.ipynb  # Start with RAG prototype
jupyter notebook L3/L3.ipynb  # Learn Airflow basics
jupyter notebook L5/L5.ipynb  # Advanced scheduling
```

### Access Airflow UI
- **URL**: `http://localhost:8080`
- **Username**: `airflow`
- **Password**: `airflow`

---

## 📊 Key Features

### 1. **Embedded Weaviate**
- Local vector database
- No external server required
- Persistence in `tmp/weaviate/`

### 2. **FastEmbed Integration**
- Model: `BAAI/bge-small-en-v1.5`
- Fast embedding generation
- Optimized for semantic search

### 3. **Airflow TaskFlow API**
- Python decorators (`@dag`, `@task`)
- Type-safe task dependencies
- XCom for data passing

### 4. **Data-Aware Scheduling**
- Asset-based triggers
- Event-driven workflows
- Automatic downstream execution

---

## 🔗 Resources

### Airflow Documentation
- [Airflow TaskFlow API](https://www.astronomer.io/docs/learn/airflow-decorators/)
- [Scheduling in Airflow](https://www.astronomer.io/docs/learn/scheduling-in-airflow/)
- [Assets and Data-Aware Scheduling](https://www.astronomer.io/docs/learn/airflow-datasets/)
- [DAG Parameters](https://www.astronomer.io/docs/learn/airflow-params/)
- [Airflow Context](https://www.astronomer.io/docs/learn/airflow-context/)

### Vector Databases & Embeddings
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [FastEmbed Overview](https://qdrant.github.io/fastembed/)
- [Vector Databases Course](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)
- [Multimodal Search and RAG](https://www.deeplearning.ai/short-courses/building-multimodal-search-and-rag/)

### Installation Guides
- [Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- [Astro CLI for Local Development](https://www.astronomer.io/docs/astro/cli/get-started-cli)
- [Course GitHub Repo](https://github.com/astronomer/orchestrating-workflows-for-genai-deeplearning-ai)

---

## 💡 Use Cases

This pattern is ideal for:

- 📚 **Document Search Systems** - Books, papers, documentation
- 🤖 **RAG-based Chatbots** - With continuously updated knowledge
- 📊 **Automated Embedding Pipelines** - Process new content automatically
- 🔄 **Event-driven AI Workflows** - Trigger downstream tasks on data updates
- 🏢 **Enterprise Knowledge Bases** - Semantic search over company documents

---

## 🐛 Known Issues & Notes

### Helper File Duplication ⚠️
Both `L2/helper.py` and `L3/helper.py` contain identical code (`suppress_output()` function). Consider extracting to a shared utility module.

### Weaviate Persistence
The embedded Weaviate instance stores data in `L2/tmp/weaviate/`. This is cleared when the environment resets.

### DAG Updates
Changes to DAG files may take up to 30 seconds to appear in the Airflow UI.

### Session Timeout
The Airflow UI may show "504 Gateway Timeout" after 2 hours or 25 minutes of inactivity. Refresh the notebook and regenerate the UI link.

---

## 📈 Next Steps

1. **Complete all lessons** in order (L2 → L3 → L5)
2. **Experiment** with custom book descriptions
3. **Modify DAG parameters** for different queries
4. **Set up local Airflow** environment for practice
5. **Explore advanced features**: sensors, branching, dynamic task mapping

---

## 📝 License & Attribution

Course content from [DeepLearning.AI](https://www.deeplearning.ai/) in partnership with [Astronomer](https://www.astronomer.io/).

---

**Happy Orchestrating! 🚀**
