# RAG with LlamaIndex Tutorial Series

This repository contains a comprehensive tutorial series on building Retrieval-Augmented Generation (RAG) systems using LlamaIndex. The project demonstrates advanced RAG techniques including router query engines, tool calling, and agent reasoning loops.

**✨ New:** Complete [Python code examples](#-code-examples-for-advanced-concepts) for all advanced concepts including dual indexing, query routing, agent reasoning, and more!

## 📁 Project Structure

```
rag_with_llamaindex/
├── README.md                           # This file
├── router_query_engine/                # Lesson 1: Router Query Engine
│   ├── L1_Router_Engine.ipynb         # Main tutorial notebook
│   ├── metagpt.pdf                    # Sample document (16MB)
│   ├── requirements.txt               # Python dependencies
│   ├── helper.py                      # Environment utilities
│   └── utils.py                       # Router engine utilities
├── tool_calling/                       # Lesson 2: Tool Calling
│   ├── L2_Tool_Calling.ipynb         # Main tutorial notebook
│   └── helper.py                      # Environment utilities
├── agent_reasoning_loop/              # Lesson 3: Agent Reasoning Loop
│   └── L3_Building_an_Agent_Reasoning_Loop.ipynb
└── multi_doc_agent/                   # Future lesson (currently empty)
```

## 🎯 Tutorial Overview

This tutorial series progresses from basic RAG concepts to advanced agent-based systems:

### Lesson 1: Router Query Engine
**Location**: `router_query_engine/`

- **Objective**: Learn to build intelligent query routing systems
- **Key Concepts**: 
  - Summary vs Vector-based retrieval
  - Router query engines with LLM selectors
  - Multi-index architectures
- **Technologies**: LlamaIndex, OpenAI, Sentence Splitting
- **Sample Document**: MetaGPT research paper analysis

### Lesson 2: Tool Calling
**Location**: `tool_calling/`

- **Objective**: Master function calling and tool integration
- **Key Concepts**:
  - Function tool creation and registration
  - Auto-retrieval tools
  - LLM-driven tool selection
- **Technologies**: FunctionTool, OpenAI Function Calling
- **Features**: Custom mathematical functions and document retrieval tools

### Lesson 3: Agent Reasoning Loop
**Location**: `agent_reasoning_loop/`

- **Objective**: Build sophisticated agent systems with reasoning capabilities
- **Key Concepts**:
  - Agent workers and runners
  - Multi-step reasoning processes
  - Tool orchestration
- **Technologies**: FunctionCallingAgentWorker, AgentRunner
- **Features**: Complex query processing with multiple tool calls

## 🛠️ Technologies Used

### Core Dependencies
- **LlamaIndex**: 0.10.27 - Core RAG framework
- **OpenAI**: GPT-3.5-turbo for LLM, text-embedding-ada-002 for embeddings
- **Python-dotenv**: 1.0.0 - Environment variable management

### Key LlamaIndex Components
- `SimpleDirectoryReader` - Document loading
- `SentenceSplitter` - Text chunking
- `SummaryIndex` & `VectorStoreIndex` - Dual indexing strategy
- `RouterQueryEngine` - Intelligent query routing
- `FunctionTool` - Tool creation and registration
- `FunctionCallingAgentWorker` - Agent implementation

## 📊 Key Features

### 1. **Intelligent Query Routing**
- Automatically routes queries to appropriate retrieval methods
- Uses LLM-based selectors for optimal tool choice
- Supports both summary and vector-based retrieval

### 2. **Tool Integration**
- Custom function tools for mathematical operations
- Auto-retrieval tools for document access
- Seamless LLM-driven tool calling

### 3. **Agent Reasoning**
- Multi-step reasoning processes
- Tool orchestration and coordination
- Memory management for conversation context

### 4. **Document Processing**
- PDF document loading and parsing
- Sentence-based text chunking (1024 tokens)
- Dual indexing for comprehensive retrieval

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.9+ recommended
python --version

# Install dependencies
pip install -r router_query_engine/requirements.txt
```

### Environment Setup
1. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

2. Ensure the MetaGPT PDF is available in the router_query_engine directory

### Running the Tutorials

#### Lesson 1: Router Query Engine
```bash
cd router_query_engine
jupyter notebook L1_Router_Engine.ipynb
```

#### Lesson 2: Tool Calling
```bash
cd tool_calling
jupyter notebook L2_Tool_Calling.ipynb
```

#### Lesson 3: Agent Reasoning Loop
```bash
cd agent_reasoning_loop
jupyter notebook L3_Building_an_Agent_Reasoning_Loop.ipynb
```

## 📚 Learning Path

### Beginner → Advanced Progression

1. **Start with Router Query Engine** (Lesson 1)
   - Understand basic RAG concepts
   - Learn about different retrieval strategies
   - Master query routing

2. **Progress to Tool Calling** (Lesson 2)
   - Build custom tools
   - Integrate function calling
   - Understand tool orchestration

3. **Advance to Agent Reasoning** (Lesson 3)
   - Create sophisticated agents
   - Implement multi-step reasoning
   - Master complex tool coordination

## 🔧 Utility Functions

### Environment Management
```python
from helper import get_openai_api_key, load_env
```

### Router Engine Setup
```python
from utils import get_router_query_engine
```

### Document Tools
```python
from utils import get_doc_tools
```

## 📈 Sample Use Cases

### 1. **Research Paper Analysis**
- Route queries to summary or detailed retrieval
- Extract specific information from MetaGPT paper
- Generate comprehensive summaries

### 2. **Mathematical Operations**
- Custom function tools for calculations
- LLM-driven mathematical reasoning
- Tool chaining for complex operations

### 3. **Multi-Step Reasoning**
- Agent-based query processing
- Tool orchestration for complex tasks
- Memory management for conversation context

## 🎓 Key Learning Outcomes

After completing this tutorial series, you will be able to:

- ✅ Build intelligent RAG systems with query routing
- ✅ Create and integrate custom tools with LLMs
- ✅ Implement sophisticated agent reasoning loops
- ✅ Handle complex document processing workflows
- ✅ Design multi-index architectures
- ✅ Master LlamaIndex's advanced features

## 🔍 Advanced Concepts Covered

Below are the key advanced concepts demonstrated in this tutorial series. **See the [Code Examples](#-code-examples-for-advanced-concepts) section below for complete implementations.**

- **Dual Indexing**: Summary and vector-based retrieval
- **Query Routing**: LLM-driven tool selection
- **Tool Orchestration**: Coordinated tool calling
- **Agent Reasoning**: Multi-step problem solving
- **Memory Management**: Conversation context preservation
- **Async Processing**: Efficient document handling

## 💻 Code Examples for Advanced Concepts

### 1. Dual Indexing: Summary and Vector-Based Retrieval

Create both summary and vector indexes over the same data for different query types:

```python
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

# Parse documents into nodes
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# Create dual indexes
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

# Create query engines with different strategies
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()
```

**When to use:**
- **Summary Index**: High-level questions, document summaries, broad overviews
- **Vector Index**: Specific factual questions, detailed context retrieval

### 2. Query Routing: LLM-Driven Tool Selection

Automatically route queries to the most appropriate retrieval method:

```python
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# Define tools with descriptions for LLM selector
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to MetaGPT"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the MetaGPT paper."
    ),
)

# Create router with LLM-based selector
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)

# The LLM automatically selects the right tool
response = query_engine.query("What is the summary of the document?")
# Routes to: summary_tool

response = query_engine.query("How do agents share information?")
# Routes to: vector_tool
```

### 3. Tool Orchestration: Coordinated Tool Calling

Define custom function tools and let the LLM orchestrate their usage:

```python
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Define custom functions
def add(x: int, y: int) -> int:
    """Adds two integers together."""
    return x + y

def mystery(x: int, y: int) -> int: 
    """Mystery function that operates on top of two numbers."""
    return (x + y) * (x + y)

# Convert to tools
add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)

# LLM orchestrates tool calling
llm = OpenAI(model="gpt-3.5-turbo")
response = llm.predict_and_call(
    [add_tool, mystery_tool], 
    "Tell me the output of the mystery function on 2 and 9", 
    verbose=True
)
# Output: 121 (automatically calls mystery(2, 9))
```

### 4. Agent Reasoning: Multi-Step Problem Solving

Build agents that can perform multi-step reasoning with tool orchestration:

```python
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.llms.openai import OpenAI

# Setup LLM
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

# Create agent with multiple tools
agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

# Agent performs multi-step reasoning
response = agent.query(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

# Agent automatically:
# 1. Calls summary_tool for "agent roles"
# 2. Calls summary_tool for "communication between agents"
# 3. Synthesizes both results into coherent response
```

**Example Output:**
```
=== Calling Function ===
Calling function: summary_tool_metagpt with args: {"input": "agent roles in MetaGPT"}
=== Function Output ===
Agent roles include Product Manager, Architect, Project Manager, Engineer, QA Engineer...

=== Calling Function ===
Calling function: summary_tool_metagpt with args: {"input": "communication between agent roles"}
=== Function Output ===
Communication is structured through message pools, subscriptions...
```

### 5. Memory Management: Conversation Context Preservation

Maintain conversation history for contextual follow-up queries:

```python
# Initial query
response = agent.query("Tell me about the agent roles in MetaGPT")

# Follow-up query using conversation memory
response = agent.chat("Tell me about the evaluation datasets used.")
# Agent remembers previous context

# Another follow-up
response = agent.chat("Tell me the results over one of the above datasets.")
# Agent knows "above datasets" refers to previous response
```

**Agent Memory Tracking:**
```python
# Create a task for more control
task = agent.create_task(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

# Step through task execution
step_output = agent.run_step(task.task_id)

# Access memory
print(task.memory)  # Shows conversation history
```

### 6. Async Processing: Efficient Document Handling

Use async processing for better performance with large documents:

```python
import nest_asyncio

# Enable nested async (for Jupyter notebooks)
nest_asyncio.apply()

# Create async-enabled query engine
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,  # Enable async processing
)

# Async queries run concurrently
response = await summary_query_engine.aquery(
    "What is the summary of the document?"
)
```

**Batch Processing with Async:**
```python
import asyncio

async def process_multiple_queries(queries):
    tasks = [
        summary_query_engine.aquery(query) 
        for query in queries
    ]
    return await asyncio.gather(*tasks)

# Process multiple queries concurrently
queries = [
    "What is MetaGPT?",
    "What are the evaluation results?",
    "What are the agent roles?"
]
responses = await process_multiple_queries(queries)
```

### 7. Auto-Retrieval with Metadata Filtering

Create retrieval tools that automatically filter by metadata:

```python
from llama_index.core.vector_stores import MetadataFilters
from typing import List

# Define auto-retrieval function
def vector_query(
    query: str, 
    page_numbers: List[str]
) -> str:
    """Perform vector search with optional page filtering.
    
    Args:
        query: The search query string
        page_numbers: List of page numbers to filter by
    """
    metadata_dict = [{"key": "page_label", "value": p} for p in page_numbers]
    
    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(metadata_dict)
    )
    
    response = query_engine.query(query)
    return response

# Convert to tool
from llama_index.core.tools import FunctionTool

vector_query_tool = FunctionTool.from_defaults(fn=vector_query)

# LLM can now query specific pages
response = llm.predict_and_call(
    [vector_query_tool],
    "What are high-level results of MetaGPT on page 2?",
    verbose=True
)
```

### 8. Complete RAG Pipeline Example

Putting it all together - a complete RAG system with routing and agents:

```python
from helper import get_openai_api_key
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import nest_asyncio

# Setup
nest_asyncio.apply()
OPENAI_API_KEY = get_openai_api_key()

# Configure models
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Load and process documents
documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# Create dual indexes
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

# Setup query engines with tools
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions"
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context"
)

# Create agent
agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=Settings.llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

# Query with multi-step reasoning
response = agent.query(
    "Give me a summary of the document, "
    "then tell me about the agent roles."
)
print(response)
```

## 🤝 Contributing

This tutorial series is designed for educational purposes. Feel free to:

- Extend the examples with your own documents
- Add new tool types and functions
- Experiment with different LLM models
- Share improvements and enhancements

## 📄 License

This project is for educational purposes. Please ensure proper attribution when using any code or methodologies.

---

*Built with LlamaIndex and OpenAI - Advanced RAG Tutorial Series* 