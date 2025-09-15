# AI Agent with LangGraph Tutorial Series

This repository contains a comprehensive tutorial series on building AI agents using LangGraph. The project demonstrates the progression from simple ReAct agents to sophisticated human-in-the-loop systems with persistence and streaming capabilities.

## 📁 Project Structure

```
ai_agent_with_langgraph/
├── README.md                              # This file
├── L1/                                    # Lesson 1: Simple ReAct Agent
│   ├── Lesson_1_Student.ipynb             # Basic ReAct pattern implementation
│   └── requirements.txt                   # Dependencies for the entire series
├── L2/                                    # Lesson 2: LangGraph Components
│   └── Lesson_2_LangGraph_Components.ipynb
├── L3/                                    # Lesson 3: Agentic Search
│   └── Lesson_3_Agentic_Search.ipynb
├── L4/                                    # Lesson 4: Persistence & Streaming
│   └── Lesson_4_persistence_streaming.ipynb
└── L5/                                    # Lesson 5: Human-in-the-Loop
    ├── Lesson_5_Human_In_The_Loop.ipynb
    ├── helper.py                          # Complex multi-agent system utilities
    └── temp_test_gradio.ipynb             # Gradio UI testing
```

## 🎯 Tutorial Overview

This tutorial series progresses from foundational concepts to advanced agent architectures:

### Lesson 1: Simple ReAct Agent from Scratch
**Location**: `L1/`

- **Objective**: Build a basic ReAct (Reasoning + Acting) agent without frameworks
- **Key Concepts**:
  - ReAct pattern implementation (Thought, Action, PAUSE, Observation)
  - Custom agent class with message handling
  - Tool integration (calculator, dog weight lookup)
  - Manual control flow and action parsing
- **Technologies**: Pure Python with OpenAI API
- **Agent Capabilities**: Basic reasoning loop with custom tools

**What's New**: Foundation - introduces the core ReAct pattern from scratch

### Lesson 2: LangGraph Components  
**Location**: `L2/`

- **Objective**: Learn LangGraph framework fundamentals and component architecture
- **Key Concepts**:
  - StateGraph and typed state management
  - Node-based agent architecture
  - Conditional edges and flow control
  - Integration with Tavily search tools
  - LangGraph's state annotation system
- **Technologies**: LangGraph, LangChain, Tavily Search
- **Agent Capabilities**: Structured agent with proper state management

**What's New vs L1**: 
- Introduces **LangGraph framework** (vs. pure Python)
- **Typed state management** with `AgentState`
- **Graph-based architecture** with nodes and edges
- **Professional tooling** integration (Tavily search)

### Lesson 3: Agentic Search
**Location**: `L3/`

- **Objective**: Master advanced search capabilities and agentic search patterns
- **Key Concepts**:
  - Direct Tavily API integration
  - Search result processing and filtering
  - Context-aware search strategies
  - Multi-query search patterns
  - Search result synthesis
- **Technologies**: Tavily Client, Advanced search APIs
- **Agent Capabilities**: Sophisticated search and information retrieval

**What's New vs L2**:
- **Direct API integration** (vs. LangChain tool wrappers)
- **Advanced search patterns** and result processing
- **Multi-step search strategies**
- **Search result synthesis** and analysis

### Lesson 4: Persistence and Streaming
**Location**: `L4/`

- **Objective**: Add memory persistence and real-time streaming capabilities
- **Key Concepts**:
  - SQLite-based state persistence
  - Checkpointing and state recovery
  - Streaming responses and real-time updates
  - Conversation memory across sessions
  - State history management
- **Technologies**: LangGraph with SqliteSaver, Streaming APIs
- **Agent Capabilities**: Persistent conversations with streaming

**What's New vs L3**:
- **State persistence** with SQLite checkpointing
- **Memory across sessions** and conversation continuity
- **Streaming capabilities** for real-time responses
- **State history** tracking and recovery

### Lesson 5: Human-in-the-Loop
**Location**: `L5/`

- **Objective**: Build sophisticated multi-agent systems with human intervention
- **Key Concepts**:
  - Complex multi-node agent workflows
  - Human approval points and intervention
  - Essay writing with plan/draft/critique cycle
  - State modification and rollback capabilities
  - Advanced Gradio UI for agent control
  - Multi-threaded conversation management
- **Technologies**: Advanced LangGraph, Gradio UI, Complex state management
- **Agent Capabilities**: Full essay writing system with human oversight

**What's New vs L4**:
- **Multi-agent workflow** (planner, researcher, writer, critic)
- **Human intervention points** with approval workflows
- **Complex state management** with rollback capabilities
- **Advanced UI** with Gradio for agent control
- **Multi-threaded conversations** and state branching

## 🛠️ Technologies Used

### Core Dependencies
- **LangGraph**: 0.0.53 - Agent workflow framework
- **LangChain**: 0.2.0 - LLM integration and tools
- **OpenAI**: 1.30.1 - GPT-3.5/GPT-4o models
- **Tavily**: 0.3.3 - Search API integration
- **Gradio**: 4.31.3 - Web UI for human-in-the-loop

### Key LangGraph Components
- `StateGraph` - Agent workflow definition
- `SqliteSaver` - State persistence and checkpointing
- `TypedDict` - Type-safe state management
- Conditional edges - Dynamic workflow routing
- Human-in-the-loop - Interactive agent control

## 📊 Key Features by Lesson

### 1. **ReAct Pattern Foundation** (L1)
- Manual thought-action-observation loops
- Custom tool integration and parsing
- Basic conversation memory

### 2. **Framework Integration** (L2)
- Professional agent architecture
- Type-safe state management
- Tool ecosystem integration

### 3. **Advanced Search** (L3)
- Multi-strategy search approaches
- Search result synthesis
- Context-aware information retrieval

### 4. **Production Features** (L4)
- Persistent conversation memory
- Real-time streaming responses
- Session continuity and recovery

### 5. **Human Collaboration** (L5)
- Multi-agent collaborative workflows
- Human oversight and intervention
- Advanced state manipulation
- Professional UI for agent control

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.11+ recommended
python --version

# Install dependencies
pip install -r L1/requirements.txt

# For graph visualization (optional)
# macOS: brew install graphviz
# Linux: sudo apt-get install python3-dev graphviz libgraphviz-dev pkg-config
```

### Environment Setup
1. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

2. Install additional dependencies for specific lessons:
```bash
pip install aiosqlite gradio  # For L4 and L5
```

### Running the Tutorials

#### Lesson 1: Simple ReAct Agent
```bash
cd L1
jupyter notebook Lesson_1_Student.ipynb
```

#### Lesson 2: LangGraph Components
```bash
cd L2
jupyter notebook Lesson_2_LangGraph_Components.ipynb
```

#### Continue with L3, L4, L5...

## 📚 Learning Path

### Beginner → Advanced Progression

1. **Start with ReAct Basics** (L1)
   - Understand agent reasoning patterns
   - Learn tool integration
   - Master conversation flow

2. **Learn LangGraph Framework** (L2)
   - Graph-based agent architecture
   - Type-safe state management
   - Professional tool integration

3. **Master Search Integration** (L3)
   - Advanced search strategies
   - Information synthesis
   - Multi-query patterns

4. **Add Production Features** (L4)
   - State persistence and memory
   - Streaming capabilities
   - Session management

5. **Build Human Collaboration** (L5)
   - Multi-agent workflows
   - Human oversight systems
   - Advanced UI integration

## 🔧 Key Concepts by Lesson

### L1: Foundation Concepts
- ReAct pattern (Thought → Action → PAUSE → Observation)
- Manual conversation management
- Basic tool calling and response parsing

### L2: Framework Architecture
- `StateGraph` and node-based workflows
- `AgentState` with type annotations
- Conditional edges and flow control

### L3: Search Mastery
- Direct API integration vs. framework wrappers
- Search result processing and filtering
- Multi-step search strategies

### L4: Production Readiness
- `SqliteSaver` for state persistence
- Checkpointing and recovery mechanisms
- Streaming response handling

### L5: Advanced Collaboration
- Multi-node agent workflows
- Human intervention and approval points
- State modification and rollback
- Complex UI for agent management

## 📈 Sample Use Cases by Lesson

### L1: Basic Question Answering
- Simple calculations and lookups
- Basic reasoning with tools
- Manual conversation flow

### L2: Search-Enabled Assistant
- Web search integration
- Structured response generation
- Tool-based problem solving

### L3: Research Assistant
- Advanced search strategies
- Information synthesis
- Context-aware retrieval

### L4: Persistent Chatbot
- Conversation continuity
- Session memory
- Real-time streaming

### L5: Essay Writing System
- Plan → Research → Draft → Critique cycle
- Human oversight and approval
- Multi-threaded conversations

## 🎓 Key Learning Outcomes

After completing this tutorial series, you will be able to:

- ✅ Build ReAct agents from scratch
- ✅ Use LangGraph for professional agent development
- ✅ Integrate advanced search capabilities
- ✅ Implement persistent agent memory
- ✅ Create human-in-the-loop agent systems
- ✅ Build complex multi-agent workflows
- ✅ Design production-ready agent applications

## 🔍 Advanced Concepts Covered

- **ReAct Pattern**: Thought-action-observation loops
- **Graph Architecture**: Node-based agent workflows
- **State Management**: Type-safe state with persistence
- **Tool Integration**: From basic to advanced search
- **Memory Systems**: Conversation persistence and recovery
- **Human Collaboration**: Approval workflows and intervention
- **UI Integration**: Professional interfaces for agent control

## 🛡️ Best Practices Demonstrated

### 1. **Agent Design**
- Clear separation of reasoning and action
- Type-safe state management
- Robust error handling

### 2. **Tool Integration**
- Progressive complexity from basic to advanced
- Proper API integration patterns
- Search result processing

### 3. **State Management**
- Persistent conversation memory
- State rollback and recovery
- Multi-threaded conversations

### 4. **Human Integration**
- Clear approval points
- Intuitive UI design
- State modification capabilities

## 🤝 Contributing

This tutorial series demonstrates the evolution of agent capabilities. Feel free to:

- Extend examples with new tools and capabilities
- Add additional search providers
- Implement new UI patterns
- Share improvements and enhancements

## 📄 License

This project is for educational purposes. Please ensure proper attribution when using any code or methodologies.

---

*Built with LangGraph and OpenAI - Advanced AI Agent Tutorial Series*
