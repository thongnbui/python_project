# Orchestrating Workflows for GenAI Applications with Apache Airflow

## 📚 Overview

This directory contains coursework from the DeepLearning.AI course: **"Orchestrating Workflows for GenAI Applications"**. The course demonstrates how to build and automate a RAG (Retrieval-Augmented Generation) system using Apache Airflow, vector databases, and embeddings.

**✨ Updated:** Now includes complete lessons 4, 6 & [7](#lesson-7-prepare-to-fail-🛡️) with full Python code examples for RAG pipelines, dynamic task mapping, and production-ready error handling!

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
├── L4/                    # Lesson 4: Notebook to Pipeline
│   └── L4.ipynb          # Transform RAG prototype to DAGs
├── L5/                    # Lesson 5: Scheduling & Parameters
│   └── L5.ipynb          # Time-based & data-aware scheduling
├── L6/                    # Lesson 6: Dynamic Task Mapping
│   └── L6.ipynb          # Make pipeline adaptable with parallel tasks
├── L7/                    # Lesson 7: Prepare to Fail
│   └── L7.ipynb          # Error handling, retries, and callbacks
└── README.md             # This file
```

---

## 🎯 Learning Objectives

- Build a RAG prototype with vector databases
- Orchestrate GenAI workflows with Apache Airflow
- Implement time-based and data-aware scheduling
- Use Airflow TaskFlow API and DAG patterns
- Master dynamic task mapping for parallel processing
- Configure retries, trigger rules, and failure callbacks
- Implement robust error handling and monitoring
- Create production-ready, resilient, scalable AI pipelines

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

#### Step-by-Step Implementation

**1. Import Libraries and Setup**
```python
import os
import json
from fastembed import TextEmbedding
import weaviate
from weaviate.classes.data import DataObject

# Configuration
COLLECTION_NAME = "Books"  # Weaviate collection name
BOOK_DESCRIPTION_FOLDER = "include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
```

**2. Create Embedded Weaviate Instance**
```python
# Instantiate local Weaviate (no external server needed)
client = weaviate.connect_to_embedded(
    persistence_data_path="tmp/weaviate",  # Local storage
)
print(f"Client is ready: {client.is_ready()}")
```

**What is Embedded Weaviate?**
- Runs directly from your Python code (no separate server)
- Perfect for prototyping and local development
- Data persists in `tmp/weaviate/` directory
- In production, you'd use a containerized Weaviate instance

**3. Create Collection (Database Schema)**
```python
# Check if collection already exists
existing_collections = client.collections.list_all()
existing_collection_names = existing_collections.keys()

if COLLECTION_NAME not in existing_collection_names:
    print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
    collection = client.collections.create(name=COLLECTION_NAME)
    print(f"Collection {COLLECTION_NAME} created successfully.")
else:
    print(f"Collection {COLLECTION_NAME} already exists. No action taken.")
    collection = client.collections.get(COLLECTION_NAME)
```

**What is a Collection?**
- A Weaviate collection = a table/schema for storing objects
- Each object has properties (metadata) + vector embedding
- Similar to a database table, but optimized for vector search

**4. Load Book Descriptions from Files**
```python
# List all .txt files in data folder
book_description_files = [
    f for f in os.listdir(BOOK_DESCRIPTION_FOLDER)
    if f.endswith('.txt')
]

# Parse each file and extract book metadata
list_of_book_data = []

for book_description_file in book_description_files:
    with open(
        os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
    ) as f:
        book_descriptions = f.readlines()
    
    # Each line format: [Index] ::: [Title] ::: [Author] ::: [Description]
    titles = [
        book_description.split(":::")[1].strip()
        for book_description in book_descriptions
    ]
    authors = [
        book_description.split(":::")[2].strip()
        for book_description in book_descriptions
    ]
    book_description_text = [
        book_description.split(":::")[3].strip()
        for book_description in book_descriptions
    ]
    
    # Create structured data
    book_descriptions = [
        {
            "title": title,
            "author": author,
            "description": description,
        }
        for title, author, description in zip(
            titles, authors, book_description_text
        )
    ]
    
    list_of_book_data.append(book_descriptions)
```

**Example Book Data Structure:**
```python
[
    {
        "title": "The Idea of the World (2019)",
        "author": "Bernardo Kastrup",
        "description": "An ontological thesis arguing for the primacy of mind over matter."
    },
    {
        "title": "Exploring the World of Lucid Dreaming (1990)",
        "author": "Stephen LaBerge",
        "description": "A practical guide to learning and enjoying lucid dreams."
    }
]
```

**5. Create Vector Embeddings**
```python
# Initialize embedding model (downloads on first use)
embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)

list_of_description_embeddings = []

# Generate embeddings for each book description
for book_data in list_of_book_data:
    book_descriptions = [book["description"] for book in book_data]
    
    # Convert text to 384-dimensional vectors
    description_embeddings = [
        list(embedding_model.embed([desc]))[0] 
        for desc in book_descriptions
    ]
    
    list_of_description_embeddings.append(description_embeddings)
```

**Why FastEmbed?**
- Fast CPU-based embedding generation (no GPU required)
- Model: `BAAI/bge-small-en-v1.5` (384 dimensions)
- Optimized for semantic similarity
- Each description → 384-dimensional vector

**6. Load Embeddings into Weaviate**
```python
# Insert books with their embeddings into Weaviate
for book_data_list, emb_list in zip(list_of_book_data, list_of_description_embeddings):
    items = []
    
    for book_data, emb in zip(book_data_list, emb_list):
        # Create data object with properties and vector
        item = DataObject(
            properties={
                "title": book_data["title"],
                "author": book_data["author"],
                "description": book_data["description"],
            },
            vector=emb  # 384-dimensional embedding
        )
        items.append(item)
    
    # Batch insert for efficiency
    collection.data.insert_many(items)

print(f"Successfully loaded {len(items)} books into Weaviate!")
```

**What's stored in Weaviate?**
- **Properties**: Metadata (title, author, description) - searchable text
- **Vector**: 384-dimensional embedding - enables semantic similarity search
- Each book is indexed for both keyword and vector search

**7. Query with Semantic Search**
```python
# User query (natural language)
query_str = "A philosophical book"

# Initialize embedding model and get collection
embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)
collection = client.collections.get(COLLECTION_NAME)

# Convert query to embedding (same 384-dimensional space)
query_emb = list(embedding_model.embed([query_str]))[0]

# Perform vector similarity search
results = collection.query.near_vector(
    near_vector=query_emb,
    limit=1,  # Return top 1 most similar book
)

# Display results
for result in results.objects:
    print(f"You should read: {result.properties['title']} by {result.properties['author']}")
    print("Description:")
    print(result.properties["description"])
```

**Example Output:**
```
You should read: The Idea of the World (2019) by Bernardo Kastrup
Description:
An ontological thesis arguing for the primacy of mind over matter.
```

**How Semantic Search Works:**
1. Query text → embedding vector (384 dimensions)
2. Compare query vector to all book vectors using cosine similarity
3. Return books with highest similarity scores
4. Understands meaning, not just keywords!

**Why This Works:**
- Query "philosophical book" matches "ontological thesis" semantically
- No keyword overlap needed - understands concepts
- Vector space captures meaning relationships
- Similar embeddings = similar meaning

#### Complete Data Flow

```
📄 Text Files (book_descriptions_1.txt, book_descriptions_2.txt)
    ↓ Parse with split(":::")
📊 Structured Data [{title, author, description}, ...]
    ↓ FastEmbed (BAAI/bge-small-en-v1.5)
🔢 Vector Embeddings [384-dimensional arrays]
    ↓ insert_many()
💾 Weaviate Collection (Books)
    ├─ Properties: {title, author, description}
    └─ Vectors: [384-dimensional embeddings]
    ↓ query.near_vector()
🔍 Semantic Search Results (ranked by similarity)
```

#### Key Takeaways

✅ **Embedded Weaviate** - Perfect for local development and prototyping

✅ **FastEmbed** - Efficient CPU-based embeddings without GPU

✅ **Semantic Search** - Find similar content by meaning, not keywords

✅ **Vector + Metadata** - Store both embeddings and searchable properties

✅ **Batch Operations** - Use `insert_many()` for better performance

✅ **RAG Foundation** - This is the retrieval component for RAG systems

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

### **Lesson 4: Turning Your Notebook Into a Pipeline** 🔄

**Key Concepts**:
- Converting RAG prototype to production DAGs
- Creating two complementary DAGs (fetch and query)
- Using Airflow Hooks and Connections
- Weaviate Hook for vector database integration
- Task skeleton approach (define structure, then implement)

**What it does**:
Transforms the L2 RAG prototype into two production-ready DAGs:

#### 1. **`fetch_data` DAG** - Data Ingestion Pipeline
**Tasks**:
1. **`create_collection_if_not_exists`** - Initialize Weaviate collection
2. **`list_book_description_files`** - Scan `/include/data/` for `.txt` files
3. **`transform_book_description_files`** - Parse text files into structured data
4. **`create_vector_embeddings`** - Generate embeddings with FastEmbed
5. **`load_embeddings_to_vector_db`** - Insert data into Weaviate

**Task Dependencies**:
```python
chain(
    _create_collection_if_not_exists,
    _load_embeddings_to_vector_db  # Implicitly waits for all upstream tasks
)
```

#### 2. **`query_data` DAG** - Query Pipeline
**Task**:
- **`search_vector_db_for_a_book`** - Semantic search using query string

**Key Technical Details**:

**Airflow Connections**:
- Uses `my_weaviate_conn` connection to securely store Weaviate credentials
- Connection defined as environment variable:
```python
AIRFLOW_CONN_MY_WEAVIATE_CONN='{
    "conn_type":"weaviate",
    "host":"localhost",
    "port":"8081",
    "extra":{
        "token":"adminkey",
        "grpc_port":"50051",
        "grpc_host":"localhost",
        "grpc_secure":"False",
        "http_secure":"False"
    }
}'
```

**Weaviate Hook**:
```python
from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

hook = WeaviateHook("my_weaviate_conn")
client = hook.get_conn()
```

**Data Flow**:
```
Text Files → List Files → Transform → Create Embeddings → Load to Weaviate
                                                              ↓
                                                    Query for Recommendations
```

**Development Approach**:
1. **Step 1**: Create DAG structure with empty task stubs (pass statements)
2. **Step 2**: Fill in complete implementation for each task
3. **Step 3**: Test in Airflow UI with manual triggers

**File Locations**:
- DAG files written to: `../../dags/fetch_data.py` and `../../dags/query_data.py`
- Book descriptions read from: `/home/jovyan/include/data/`

**Complete Code Examples**:

**fetch_data.py** - Complete Implementation:
```python
from airflow.sdk import chain, dag, task 

COLLECTION_NAME = "Books" 
BOOK_DESCRIPTION_FOLDER = "/home/jovyan/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

@dag
def fetch_data():

    @task
    def create_collection_if_not_exists() -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.collections.create(name=COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        import os
        
        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER)
            if f.endswith('.txt')
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_files: list) -> list:
        import os

        list_of_book_data = []
        
        for book_description_file in book_description_files:
            with open(
                os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
            ) as f:
                book_descriptions = f.readlines()
            
            # Parse format: ID:::Title:::Author:::Description
            titles = [
                book_description.split(":::")[1].strip()
                for book_description in book_descriptions
            ]
            authors = [
                book_description.split(":::")[2].strip()
                for book_description in book_descriptions
            ]
            book_description_text = [
                book_description.split(":::")[3].strip()
                for book_description in book_descriptions
            ]
            
            book_descriptions = [
                {
                    "title": title,
                    "author": author,
                    "description": description,
                }
                for title, author, description in zip(
                    titles, authors, book_description_text
                )
            ]
        
            list_of_book_data.append(book_descriptions)

        return list_of_book_data

    _transform_book_description_files = transform_book_description_files(
        book_description_files=_list_book_description_files
    )

    @task
    def create_vector_embeddings(list_of_book_data: list) -> list:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)  
        
        list_of_description_embeddings = []
        
        for book_data in list_of_book_data:
            book_descriptions = [book["description"] for book in book_data]
            description_embeddings = [
                list(map(float, next(embedding_model.embed([desc])))) 
                for desc in book_descriptions
            ]
        
            list_of_description_embeddings.append(description_embeddings)

        return list_of_description_embeddings

    _create_vector_embeddings = create_vector_embeddings(
        list_of_book_data=_transform_book_description_files
    )

    @task
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from weaviate.classes.data import DataObject

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)

        for book_data_list, emb_list in zip(list_of_book_data, list_of_description_embeddings):
            items = []
            
            for book_data, emb in zip(book_data_list, emb_list):
                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "author": book_data["author"],
                        "description": book_data["description"],
                    },
                    vector=emb
                )
                items.append(item)
            
            # Batch insert for efficiency
            collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(
        _create_collection_if_not_exists,
        _load_embeddings_to_vector_db
    )

fetch_data()
```

**query_data.py** - Complete Implementation:
```python
from airflow.sdk import dag, task  

COLLECTION_NAME = "Books"  
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

@dag
def query_data():

    @task
    def search_vector_db_for_a_book(query_str: str) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from fastembed import TextEmbedding

        # Connect to Weaviate
        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        # Initialize embedding model
        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)  
        collection = client.collections.get(COLLECTION_NAME)
        
        # Create query embedding
        query_emb = list(embedding_model.embed([query_str]))[0]
        
        # Perform semantic search
        results = collection.query.near_vector(
            near_vector=query_emb,
            limit=1,
        )
        
        # Print results
        for result in results.objects:
            print(f"You should read: {result.properties['title']} by {result.properties['author']}")
            print("Description:")
            print(result.properties["description"])

    search_vector_db_for_a_book(query_str="A philosophical book")

query_data()
```

**Writing DAGs from Notebook**:
```python
# Use %%writefile magic command in Jupyter to create DAG files
%%writefile ../../dags/fetch_data.py 
# ... DAG code here ...

%%writefile ../../dags/query_data.py 
# ... DAG code here ...
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

### **Lesson 6: Make the Pipeline Adaptable** 🔀

**Key Concepts**:
- Dynamic Task Mapping for parallel processing
- Scaling pipelines based on input data
- `.expand()` and `.partial()` methods
- Avoiding single point of failure in data processing
- Adaptable workflows that handle variable input sizes

**Problem Statement**:
In Lesson 4's `fetch_data` DAG, the `transform_book_description_files` task processes all text files sequentially in a single task. This creates two issues:
1. **Single Point of Failure:** If one file has a formatting error, the entire task fails and must be rerun for all files
2. **Inefficiency:** Files are processed sequentially instead of in parallel

**Solution: Dynamic Task Mapping**

Dynamic Task Mapping allows Airflow to automatically create multiple parallel task instances based on input data at runtime.

**Key Methods**:
- **`.expand()`**: Creates multiple task instances, one per input value
- **`.partial()`**: Sets constant arguments that remain the same across all mapped instances

#### Simple Dynamic Task Mapping Example

```python
from airflow.sdk import dag, task 

@dag
def simple_mapping():

    @task
    def get_numbers():
        import random
        return [_ for _ in range(random.randint(0, 3))]  # Returns [0, 1, 2]

    _get_numbers = get_numbers()

    @task
    def mapped_task_one(my_constant_arg: int, my_changing_arg: int):
        return my_constant_arg + my_changing_arg

    # .partial() sets constant argument (10 for all instances)
    # .expand() creates separate task for each number
    _mapped_task_one = mapped_task_one.partial(
        my_constant_arg=10
    ).expand(my_changing_arg=_get_numbers)
    
    # Result: Creates 3 tasks: 10+0=10, 10+1=11, 10+2=12

simple_mapping()
```

**Chaining Mapped Tasks:**
```python
@dag
def simple_mapping():
    
    @task
    def get_numbers():
        import random
        return [_ for _ in range(random.randint(0, 3))]

    _get_numbers = get_numbers()

    @task
    def mapped_task_one(my_constant_arg: int, my_changing_arg: int):
        return my_constant_arg + my_changing_arg

    _mapped_task_one = mapped_task_one.partial(
        my_constant_arg=10
    ).expand(my_changing_arg=_get_numbers)

    @task
    def mapped_task_two(my_cookie_number: int):
        print(f"There are {my_cookie_number} cookies in the jar!")

    # Downstream mapped task automatically maps over upstream results
    mapped_task_two.expand(my_cookie_number=_mapped_task_one)

simple_mapping()
```

#### Applying Dynamic Task Mapping to `fetch_data`

**Before (Sequential Processing):**
```python
@task
def transform_book_description_files(book_description_files: list) -> list:
    list_of_book_data = []
    
    # Loop through all files in one task
    for book_description_file in book_description_files:
        with open(os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r") as f:
            book_descriptions = f.readlines()
        
        # Process file...
        list_of_book_data.append(book_descriptions)
    
    return list_of_book_data

# Single task processes all files
_transform_book_description_files = transform_book_description_files(
    book_description_files=_list_book_description_files
)
```

**After (Parallel Processing with Dynamic Task Mapping):**
```python
@task
def transform_book_description_files(book_description_file: str) -> list:
    # Changed: Input is now a single file (str), not a list
    # Removed: Outer for-loop
    
    with open(os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r") as f:
        book_descriptions = f.readlines()
    
    titles = [
        book_description.split(":::")[1].strip()
        for book_description in book_descriptions
    ]
    authors = [
        book_description.split(":::")[2].strip()
        for book_description in book_descriptions
    ]
    book_description_text = [
        book_description.split(":::")[3].strip()
        for book_description in book_descriptions
    ]
    
    book_descriptions = [
        {
            "title": title,
            "author": author,
            "description": description,
        }
        for title, author, description in zip(titles, authors, book_description_text)
    ]
    
    # Returns data for ONE file
    return book_descriptions

# .expand() creates one task per file - parallel processing!
_transform_book_description_files = transform_book_description_files.expand(
    book_description_file=_list_book_description_files
)
```

**Similarly for Embeddings:**
```python
@task
def create_vector_embeddings(book_data: list) -> list:
    # Changed: Input is single book_data, not list of book_data
    # Removed: Outer for-loop
    
    from fastembed import TextEmbedding
    embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)
    
    book_descriptions = [book["description"] for book in book_data]
    description_embeddings = [
        list(map(float, next(embedding_model.embed([desc]))))
        for desc in book_descriptions
    ]
    
    # Returns embeddings for ONE file
    return description_embeddings

# .expand() automatically maps over transformed files
_create_vector_embeddings = create_vector_embeddings.expand(
    book_data=_transform_book_description_files
)
```

**Complete Updated `fetch_data` DAG:**
```python
from airflow.sdk import chain, dag, task, Asset
from pendulum import datetime

COLLECTION_NAME = "Books"
BOOK_DESCRIPTION_FOLDER = "/home/jovyan/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly"
)
def fetch_data():

    @task
    def create_collection_if_not_exists() -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.collections.create(name=COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        import os
        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER) if f.endswith(".txt")
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_file: str) -> list:
        import os
        
        with open(
            os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
        ) as f:
            book_descriptions = f.readlines()

        titles = [desc.split(":::")[1].strip() for desc in book_descriptions]
        authors = [desc.split(":::")[2].strip() for desc in book_descriptions]
        book_description_text = [desc.split(":::")[3].strip() for desc in book_descriptions]

        book_descriptions = [
            {"title": title, "author": author, "description": description}
            for title, author, description in zip(titles, authors, book_description_text)
        ]

        return book_descriptions

    # Dynamic task mapping - creates one task per file
    _transform_book_description_files = transform_book_description_files.expand(
        book_description_file=_list_book_description_files
    )

    @task
    def create_vector_embeddings(book_data: list) -> list:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)
        book_descriptions = [book["description"] for book in book_data]
        description_embeddings = [
            list(map(float, next(embedding_model.embed([desc]))))
            for desc in book_descriptions
        ]

        return description_embeddings

    # Automatically creates one task per transformed file
    _create_vector_embeddings = create_vector_embeddings.expand(
        book_data=_transform_book_description_files
    )

    @task(outlets=[Asset("my_book_vector_data")])
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from weaviate.classes.data import DataObject

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)

        for book_data_list, emb_list in zip(
            list_of_book_data, list_of_description_embeddings
        ):
            items = []
            for book_data, emb in zip(book_data_list, emb_list):
                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "author": book_data["author"],
                        "description": book_data["description"],
                    },
                    vector=emb,
                )
                items.append(item)
            
            collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(_create_collection_if_not_exists, _load_embeddings_to_vector_db)

fetch_data()
```

**Benefits of Dynamic Task Mapping:**

1. **Parallel Processing** 🚀
   - Multiple files processed simultaneously
   - Faster overall pipeline execution
   - Better resource utilization

2. **Fault Isolation** 🛡️
   - If one file fails, others continue processing
   - Only failed file needs reprocessing
   - Improved reliability

3. **Scalability** 📈
   - Automatically adapts to any number of files
   - No code changes needed for more/fewer files
   - Dynamic at runtime

4. **Visibility** 👁️
   - Each file gets its own task in Airflow UI
   - Easy to identify which file failed
   - Better monitoring and debugging

**Adding Custom Book Descriptions:**
```python
# Add your own book description file
my_book_description = """0 ::: The Idea of the World (2019) ::: Bernardo Kastrup ::: An ontological thesis arguing for the primacy of mind over matter.
1 ::: Exploring the World of Lucid Dreaming (1990) ::: Stephen LaBerge ::: A practical guide to learning and enjoying lucid dreams.
"""

my_book_description_file_name = "my_descs_1.txt"

# Write to include/data directory
with open(f"../../include/data/{my_book_description_file_name}", 'w') as f:
    f.write(my_book_description)

# Next fetch_data run will automatically process this new file!
```

**Configuration Options:**

```python
@task(
    max_active_tis_per_dag=10,      # Limit concurrent tasks across all DAG runs
    max_active_tis_per_dagrun=5     # Limit concurrent tasks per single DAG run
)
def my_mapped_task(item):
    # Process item
    pass
```

**Default Limits:**
- Maximum mapped task instances: 1024 (configurable via `AIRFLOW__CORE__MAX_MAP_LENGTH`)

---

### **Lesson 7: Prepare to Fail** 🛡️

**Key Concepts**:
- Configuring automatic task retries
- Understanding trigger rules for task execution
- Implementing failure callbacks for notifications
- Building resilient production pipelines
- Handling transient vs. permanent failures

**Overview**:
Production pipelines need to handle failures gracefully. Lesson 7 teaches you how to make your DAGs resilient through retries, trigger rules, and custom callbacks that notify you when things go wrong.

#### Why Failure Handling Matters

Even well-designed pipelines can experience failures:
- **Transient failures**: Network timeouts, temporary API unavailability, resource contention
- **Permanent failures**: Code bugs, invalid data, configuration errors
- **Partial failures**: Some mapped tasks succeed while others fail

Without proper failure handling, one small issue can halt your entire pipeline.

#### 7.1. Testing Failures

First, intentionally create a failure to understand behavior:

```python
@task
def create_collection_if_not_exists() -> None:
    print(10/0)  # Intentional error - division by zero!
    # Rest of the code won't execute...
```

**What happens**: The task immediately fails with `ZeroDivisionError`, and downstream tasks are skipped.

#### 7.2. DAG-Level Retries

Configure retries for all tasks in a DAG using `default_args`:

```python
from pendulum import datetime, duration

@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly",
    default_args={
        "retries": 1,                          # Retry once on failure
        "retry_delay": duration(seconds=10)     # Wait 10 seconds between retries
    }
)
def fetch_data():
    # All tasks inherit these retry settings
    pass
```

**How it works**:
1. Task fails
2. Airflow waits 10 seconds
3. Task retries automatically
4. If retry succeeds, pipeline continues
5. If retry fails, task marked as failed

**Best for**: Transient failures like temporary network issues

#### 7.3. Task-Level Retries

Override DAG defaults for specific tasks that need more retries:

```python
@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly",
    default_args={
        "retries": 1,                          # Default: 1 retry
        "retry_delay": duration(seconds=10)
    }
)
def fetch_data():

    @task(
        retries=5,                             # Override: 5 retries for this task
        retry_delay=duration(seconds=2)        # Override: Wait only 2 seconds
    )
    def create_collection_if_not_exists() -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        
        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()  # Might have temporary connection issues
        
        # Create collection logic...
```

**When to use**:
- Tasks connecting to external services (databases, APIs)
- Tasks that might experience resource contention
- Critical tasks that are worth retrying more aggressively

#### 7.4. Trigger Rules

Control when a task should run based on upstream task states.

**Default behavior** (`all_success`): Task runs only if all upstream tasks succeed.

**Problem scenario**:
```python
# With default trigger rule:
# If create_collection_if_not_exists fails after all retries,
# load_embeddings_to_vector_db will be SKIPPED (never runs)

chain(_create_collection_if_not_exists, _load_embeddings_to_vector_db)
```

**Solution**: Use `all_done` trigger rule to run regardless of upstream success/failure:

```python
@task(
    outlets=[Asset("my_book_vector_data")],
    trigger_rule="all_done"  # Run whether upstream tasks succeed or fail
)
def load_embeddings_to_vector_db(
    list_of_book_data: list, 
    list_of_description_embeddings: list
) -> None:
    # This task runs even if some upstream tasks failed
    # You can implement logic to handle partial data
    pass
```

**Common Trigger Rules**:

| Trigger Rule | When Task Runs |
|--------------|----------------|
| `all_success` | All upstream tasks succeeded (default) |
| `all_failed` | All upstream tasks failed |
| `all_done` | All upstream tasks finished (success or failure) |
| `one_success` | At least one upstream task succeeded |
| `one_failed` | At least one upstream task failed |
| `none_failed` | No upstream tasks failed (success or skipped) |
| `none_skipped` | No upstream tasks were skipped |

**Use case for `all_done`**:
- Cleanup tasks that should always run
- Tasks that can process partial data
- Notification tasks that report status regardless of pipeline success

#### 7.5. Failure Callbacks

Get notified when tasks or DAGs fail using callback functions:

**Define a callback function:**
```python
def _my_callback_func(context):
    """
    Callback function executed when a task or DAG fails.
    
    Args:
        context: Dictionary with execution context information
    """
    task_instance = context["task_instance"]
    dag_run = context["dag_run"]
    
    print(
        f"CALLBACK: Task {task_instance.task_id} "
        f"failed in DAG {dag_run.dag_id} at {dag_run.start_date}"
    )
    
    # In production, you would:
    # - Send email notification
    # - Post to Slack/Teams
    # - Create PagerDuty incident
    # - Log to monitoring system
```

**Apply callback at DAG level:**
```python
@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly",
    default_args={
        "retries": 1,
        "retry_delay": duration(seconds=10),
        "on_failure_callback": _my_callback_func,  # Task-level callback
    },
    on_failure_callback=_my_callback_func  # DAG-level callback
)
def fetch_data():
    pass
```

**Callback types**:
- `on_failure_callback`: Runs when task/DAG fails (after all retries exhausted)
- `on_success_callback`: Runs when task/DAG succeeds
- `on_retry_callback`: Runs each time a task retries
- `on_execute_callback`: Runs before task execution starts

**Context dictionary contents**:
```python
context = {
    "task_instance": ...,  # Current task instance
    "dag_run": ...,       # Current DAG run
    "task": ...,          # Task object
    "dag": ...,           # DAG object
    "execution_date": ..., # Execution date
    "run_id": ...,        # Unique run identifier
    "params": ...,        # DAG parameters
    # ... and more
}
```

#### Complete Example with All Features

```python
from airflow.sdk import chain, dag, task, Asset
from pendulum import datetime, duration

COLLECTION_NAME = "Books"
BOOK_DESCRIPTION_FOLDER = "/home/jovyan/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def _my_callback_func(context):
    """Notify on failure - in production, send to Slack/Email/PagerDuty"""
    task_instance = context["task_instance"]
    dag_run = context["dag_run"]
    print(
        f"⚠️ ALERT: Task {task_instance.task_id} "
        f"failed in DAG {dag_run.dag_id} at {dag_run.start_date}"
    )

@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly",
    default_args={
        "retries": 1,                          # Default: 1 retry for all tasks
        "retry_delay": duration(seconds=10),    # Default: Wait 10 seconds
        "on_failure_callback": _my_callback_func,  # Notify on task failures
    },
    on_failure_callback=_my_callback_func      # Notify on DAG-level failures
)
def fetch_data():

    @task(
        retries=5,                             # Override: More retries for DB task
        retry_delay=duration(seconds=2)        # Override: Shorter delay
    )
    def create_collection_if_not_exists() -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.collections.create(name=COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        import os
        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER) if f.endswith(".txt")
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_file: str) -> list:
        import os
        
        with open(
            os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
        ) as f:
            book_descriptions = f.readlines()

        titles = [desc.split(":::")[1].strip() for desc in book_descriptions]
        authors = [desc.split(":::")[2].strip() for desc in book_descriptions]
        book_description_text = [desc.split(":::")[3].strip() for desc in book_descriptions]

        return [
            {"title": title, "author": author, "description": description}
            for title, author, description in zip(titles, authors, book_description_text)
        ]

    _transform_book_description_files = transform_book_description_files.expand(
        book_description_file=_list_book_description_files
    )

    @task
    def create_vector_embeddings(book_data: list) -> list:
        from fastembed import TextEmbedding
        
        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)
        book_descriptions = [book["description"] for book in book_data]
        description_embeddings = [
            list(map(float, next(embedding_model.embed([desc]))))
            for desc in book_descriptions
        ]
        return description_embeddings

    _create_vector_embeddings = create_vector_embeddings.expand(
        book_data=_transform_book_description_files
    )

    @task(
        outlets=[Asset("my_book_vector_data")],
        trigger_rule="all_done"  # Run even if some upstream tasks failed
    )
    def load_embeddings_to_vector_db(
        list_of_book_data: list, 
        list_of_description_embeddings: list
    ) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from weaviate.classes.data import DataObject

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)

        for book_data_list, emb_list in zip(
            list_of_book_data, list_of_description_embeddings
        ):
            items = []
            for book_data, emb in zip(book_data_list, emb_list):
                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "author": book_data["author"],
                        "description": book_data["description"],
                    },
                    vector=emb,
                )
                items.append(item)
            
            collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(_create_collection_if_not_exists, _load_embeddings_to_vector_db)

fetch_data()
```

#### Best Practices for Production

**1. Retry Strategy:**
```python
# For external service calls (DB, API)
@task(retries=5, retry_delay=duration(seconds=30))

# For resource-intensive tasks
@task(retries=2, retry_delay=duration(minutes=5))

# For critical final tasks
@task(retries=10, retry_delay=duration(seconds=60))
```

**2. Exponential Backoff:**
```python
from pendulum import duration

@task(
    retries=5,
    retry_delay=duration(seconds=10),
    retry_exponential_backoff=True,  # Wait: 10s, 20s, 40s, 80s, 160s
    max_retry_delay=duration(minutes=10)  # Cap at 10 minutes
)
def my_task():
    pass
```

**3. Production Callbacks:**
```python
def notify_slack(context):
    """Send Slack notification on failure"""
    from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook
    
    slack_hook = SlackWebhookHook(slack_webhook_conn_id="slack_conn")
    slack_hook.send(
        text=f"❌ DAG {context['dag'].dag_id} failed!",
        attachments=[{
            "color": "danger",
            "fields": [
                {"title": "Task", "value": context['task_instance'].task_id},
                {"title": "Execution Date", "value": str(context['execution_date'])},
                {"title": "Log URL", "value": context['task_instance'].log_url},
            ]
        }]
    )

def notify_email(context):
    """Send email notification"""
    from airflow.providers.email.operators.email import EmailOperator
    # Email notification logic...
```

**4. Monitoring Integration:**
```python
def log_to_datadog(context):
    """Log failure metrics to Datadog"""
    from datadog import statsd
    
    statsd.increment('airflow.task.failure',
                    tags=[
                        f"dag_id:{context['dag'].dag_id}",
                        f"task_id:{context['task_instance'].task_id}"
                    ])
```

#### Testing Failure Scenarios

**Test 1: Intentional Failure**
```python
# Add this line to make task fail
print(10/0)

# Observe in Airflow UI:
# - Task turns red (failed)
# - Automatic retries occur
# - Callback executes after final failure
```

**Test 2: Fix and Retry**
```python
# Remove the error line
# print(10/0)  # Commented out

# In Airflow UI:
# - Click "Clear" on failed task
# - Task reruns with fixed code
# - Pipeline continues normally
```

**Test 3: Partial Failure with Dynamic Tasks**
```python
# Make one file invalid to test partial failure
# Observe: Other files continue processing
# Only the problematic file's tasks fail
```

#### Key Takeaways

✅ **Always configure retries** for production DAGs - transient failures are common

✅ **Use appropriate retry delays** - too short wastes resources, too long delays recovery

✅ **Implement callbacks** for critical DAGs - get notified immediately on failures

✅ **Choose trigger rules wisely** - `all_done` for cleanup tasks, `all_success` for data quality

✅ **Test failure scenarios** - intentionally break things to verify resilience

✅ **Monitor and iterate** - adjust retry strategies based on actual failure patterns

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
jupyter notebook L4/L4.ipynb  # Convert notebook to pipeline
jupyter notebook L5/L5.ipynb  # Advanced scheduling
jupyter notebook L6/L6.ipynb  # Dynamic task mapping
jupyter notebook L7/L7.ipynb  # Error handling and retries
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
- [Connections & Hooks](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/connections.html)
- [Airflow Hooks](https://www.astronomer.io/docs/learn/what-is-a-hook/)
- [Manage Connections in Apache Airflow](https://www.astronomer.io/docs/learn/connections)
- [Custom XCom Backend Strategies](https://www.astronomer.io/docs/learn/custom-xcom-backend-strategies/)
- **[Create Dynamic Airflow Tasks](https://www.astronomer.io/docs/learn/dynamic-tasks/)** - Complete guide to dynamic task mapping
- [MAX_MAP_LENGTH Configuration](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html#max-map-length) - Configure max mapped task instances

### Vector Databases & Embeddings
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Airflow Weaviate Provider Package](https://airflow.apache.org/docs/apache-airflow-providers-weaviate/stable/index.html)
- [FastEmbed Overview](https://qdrant.github.io/fastembed/)
- [Vector Databases Course](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)
- [Multimodal Search and RAG](https://www.deeplearning.ai/short-courses/building-multimodal-search-and-rag/)

### Installation Guides
- [Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- [Astro CLI for Local Development](https://www.astronomer.io/docs/astro/cli/get-started-cli)
- [Course GitHub Repo](https://github.com/astronomer/orchestrating-workflows-for-genai-deeplearning-ai)

### Error Handling & Monitoring (Lesson 7)
- [Airflow Trigger Rules](https://www.astronomer.io/docs/learn/airflow-trigger-rules/) - Complete reference for all trigger rules
- [Manage DAG Notifications](https://www.astronomer.io/docs/learn/error-notifications-in-airflow/) - Notifier classes and notification patterns
- [Airflow Apprise Provider](https://airflow.apache.org/docs/apache-airflow-providers-apprise/stable/index.html) - Integration with 80+ notification services
- [Deploy to Astro](https://www.astronomer.io/lp/signup/?utm_source=deeplearning-ai&utm_medium=content&utm_campaign=genai-course-6-25) - Free trial for cloud deployment

---

## 💡 Use Cases

This pattern is ideal for:

- 📚 **Document Search Systems** - Books, papers, documentation (with parallel processing)
- 🤖 **RAG-based Chatbots** - With continuously updated knowledge
- 📊 **Automated Embedding Pipelines** - Process new content automatically
- 🔄 **Event-driven AI Workflows** - Trigger downstream tasks on data updates
- 🏢 **Enterprise Knowledge Bases** - Semantic search over company documents
- ⚡ **Large-scale Data Processing** - Dynamic task mapping for variable workloads
- 🔀 **Batch Processing Jobs** - Parallel processing of files, records, or data chunks
- 🛡️ **Production AI Pipelines** - With robust error handling, retries, and monitoring

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

1. **Complete all lessons** in order (L2 → L3 → L4 → L5 → L6 → L7)
2. **Experiment** with custom book descriptions in L6
3. **Test failure scenarios** from L7 in a safe environment
4. **Implement dynamic task mapping** in your own pipelines
5. **Configure retries and callbacks** for production resilience
6. **Modify DAG parameters** for different queries
7. **Set up local Airflow** environment for practice
8. **Explore advanced features**: sensors, branching, task groups, SLAs
9. **Convert your own notebooks** to production DAGs using L4 patterns
10. **Optimize parallel processing** with dynamic task mapping from L6
11. **Add monitoring and alerting** using L7 callback patterns
12. **Deploy to production** with proper error handling and observability

---

## 📝 License & Attribution

Course content from [DeepLearning.AI](https://www.deeplearning.ai/) in partnership with [Astronomer](https://www.astronomer.io/).

---

**Happy Orchestrating! 🚀**
