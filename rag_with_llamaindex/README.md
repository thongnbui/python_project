# RAG with LlamaIndex Tutorial Series

This repository contains a comprehensive tutorial series on building Retrieval-Augmented Generation (RAG) systems using LlamaIndex. The project demonstrates advanced RAG techniques including router query engines, tool calling, and agent reasoning loops.

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

- **Dual Indexing**: Summary and vector-based retrieval
- **Query Routing**: LLM-driven tool selection
- **Tool Orchestration**: Coordinated tool calling
- **Agent Reasoning**: Multi-step problem solving
- **Memory Management**: Conversation context preservation
- **Async Processing**: Efficient document handling

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