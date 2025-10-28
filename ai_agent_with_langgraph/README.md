# AI Agent with LangGraph Tutorial Series

This repository contains a comprehensive tutorial series on building AI agents using LangGraph. The project demonstrates the progression from simple ReAct agents to sophisticated human-in-the-loop systems with persistence and streaming capabilities.

**✨ New:** Complete [Python code examples](#-python-code-examples-by-lesson) for all 5 lessons - from basic ReAct patterns to advanced human-in-the-loop systems!

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

## 💻 Python Code Examples by Lesson

### Lesson 1: Simple ReAct Agent from Scratch

Build a basic ReAct agent with custom tools and manual control flow:

```python
from openai import OpenAI
import re

client = OpenAI()

class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="gpt-4o", 
            temperature=0,
            messages=self.messages
        )
        return completion.choices[0].message.content
```

**Define the ReAct Prompt:**
```python
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()
```

**Create Custom Tools:**
```python
def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return "Scottish Terriers average 20 lbs"
    elif name in "Border Collie":
        return "a Border Collies average weight is 37 lbs"
    elif name in "Toy Poodle":
        return "a toy poodles average weight is 7 lbs"
    else:
        return "An average dog weights 50 lbs"

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}
```

**Run the Agent:**
```python
abot = Agent(prompt)

# First call - agent decides action
result = abot("How much does a toy poodle weigh?")
print(result)
# Output: Thought: I should look up the average weight...
#         Action: average_dog_weight: Toy Poodle
#         PAUSE

# Execute the tool
result = average_dog_weight("Toy Poodle")

# Provide observation back to agent
next_prompt = "Observation: {}".format(result)
final_answer = abot(next_prompt)
# Output: Answer: A toy poodle weights 7 lbs
```

**Multi-Step Example:**
```python
abot = Agent(prompt)

question = """I have 2 dogs, a border collie and a scottish terrier. 
What is their combined weight"""

# Agent will make multiple tool calls
abot(question)
# Step 1: Look up Border Collie weight
abot("Observation: {}".format(average_dog_weight("Border Collie")))
# Step 2: Look up Scottish Terrier weight  
abot("Observation: {}".format(average_dog_weight("Scottish Terrier")))
# Step 3: Calculate combined weight
# Output: Answer: The combined weight is 57 lbs
```

### Lesson 2: LangGraph Components

Build structured agents with LangGraph's StateGraph and typed state management:

**Define Typed State:**
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
```

**Setup Tools:**
```python
tool = TavilySearchResults(max_results=4)
print(tool.name)  # 'tavily_search_results_json'
```

**Build Graph-Based Agent:**
```python
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        
        # Add conditional edges
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:
                print("....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(
                ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result))
            )
        print("Back to the model!")
        return {'messages': results}
```

**Create and Use Agent:**
```python
prompt = """You are a smart research assistant. Use the search engine to look up information. 
You are allowed to make multiple calls (either together or in sequence). 
Only look up information when you are sure of what you want. 
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

model = ChatOpenAI(model="gpt-3.5-turbo")
abot = Agent(model, [tool], system=prompt)

# Single query
messages = [HumanMessage(content="What is the weather in sf?")]
result = abot.graph.invoke({"messages": messages})
# Output: Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}}
#         Back to the model!

print(result['messages'][-1].content)
```

**Parallel Tool Calls:**
```python
messages = [HumanMessage(content="What is the weather in SF and LA?")]
result = abot.graph.invoke({"messages": messages})
# Output: Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}}
#         Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in Los Angeles'}}
#         Back to the model!
```

**Multi-Step Reasoning:**
```python
query = """Who won the super bowl in 2024? In what state is the winning team headquarters located? 
What is the GDP of that state? Answer each question."""

messages = [HumanMessage(content=query)]
model = ChatOpenAI(model="gpt-4o")
abot = Agent(model, [tool], system=prompt)
result = abot.graph.invoke({"messages": messages})

# Output: 
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': '2024 Super Bowl winner'}}
# Back to the model!
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'Kansas City Chiefs headquarters location'}}
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'GDP of Missouri 2024'}}
# Back to the model!

print(result['messages'][-1].content)
# 1. The Kansas City Chiefs won the Super Bowl in 2024.
# 2. The headquarters is in Kansas City, Missouri.
# 3. Missouri's GDP was approximately $356.65 billion in 2024.
```

**Visualize the Graph:**
```python
from IPython.display import Image
Image(abot.graph.get_graph().draw_png())
```

### Lesson 3: Agentic Search

Master advanced search with direct API integration and result processing:

**Direct Tavily API Integration:**
```python
from tavily import TavilyClient
import os

client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# Search with answer extraction
result = client.search(
    "What is in Nvidia's new Blackwell GPU?",
    include_answer=True
)

print(result["answer"])
```

**Basic Web Search with DuckDuckGo:**
```python
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import re

ddg = DDGS()

def search(query, max_results=6):
    try:
        results = ddg.text(query, max_results=max_results)
        return [i["href"] for i in results]
    except Exception as e:
        print(f"Exception: {e}")
        return []

city = "San Francisco"
query = f"what is the current weather in {city}? Should I travel there today?"

# Get search results
urls = search(query)
for url in urls:
    print(url)
```

**Web Scraping:**
```python
def scrape_weather_info(url):
    """Scrape content from the given URL"""
    if not url:
        return "Weather information could not be found."
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return "Failed to retrieve the webpage."

    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

# Use first search result
url = search(query)[0]
soup = scrape_weather_info(url)

# Extract relevant text
weather_data = []
for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
    text = tag.get_text(" ", strip=True)
    weather_data.append(text)

weather_data = "\n".join(weather_data)
weather_data = re.sub(r'\s+', ' ', weather_data)
print(weather_data)
```

**Agentic Search with Structured Results:**
```python
# Search with structured output
result = client.search(query, max_results=1)
data = result["results"][0]["content"]

# Pretty print JSON results
import json
from pygments import highlight, lexers, formatters

parsed_json = json.loads(data.replace("'", '"'))
formatted_json = json.dumps(parsed_json, indent=4)
colorful_json = highlight(
    formatted_json,
    lexers.JsonLexer(),
    formatters.TerminalFormatter()
)
print(colorful_json)
```

**Search Result Structure:**
```python
# Access structured weather data
location = parsed_json['location']
current = parsed_json['current']

print(f"Location: {location['name']}, {location['region']}")
print(f"Temperature: {current['temp_f']}°F")
print(f"Condition: {current['condition']['text']}")
print(f"Humidity: {current['humidity']}%")
```

### Lesson 4: Persistence and Streaming

Add memory and real-time streaming to your agents:

**Setup Persistence with SQLite:**
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Create in-memory SQLite database for checkpointing
memory = SqliteSaver.from_conn_string(":memory:")
```

**Agent with Checkpointer:**
```python
class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        
        # Compile with checkpointer for persistence
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    # ... other methods same as Lesson 2 ...
```

**Create Persistent Agent:**
```python
model = ChatOpenAI(model="gpt-4o")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)
```

**Use Thread IDs for Conversation Memory:**
```python
messages = [HumanMessage(content="What is the weather in sf?")]
thread = {"configurable": {"thread_id": "1"}}

# Stream responses
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v['messages'])
```

**Follow-up with Memory:**
```python
# Follow-up question using same thread - agent remembers context
messages = [HumanMessage(content="What about in LA?")]
thread = {"configurable": {"thread_id": "1"}}

for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
```

**Agent Remembers Previous Context:**
```python
# Ask comparative question - agent knows both cities from history
messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "1"}}

for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

# Output: Los Angeles is warmer than San Francisco...
```

**Different Thread = No Memory:**
```python
# Same question but different thread - no memory
messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "2"}}

for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

# Output: Could you please clarify what you're comparing...
```

**Token-Level Streaming:**
```python
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver

memory = AsyncSqliteSaver.from_conn_string(":memory:")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)

messages = [HumanMessage(content="What is the weather in SF?")]
thread = {"configurable": {"thread_id": "4"}}

# Stream individual tokens
async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            print(content, end="|")

# Output: The| current| weather| in| San| Francisco| is| partly| cloudy|...
```

### Lesson 5: Human-in-the-Loop

Build agents with human approval and intervention:

**Custom Message Reducer for State Modification:**
```python
from uuid import uuid4
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage

def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    """Custom reducer that allows message replacement"""
    # Assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    
    # Merge new messages with existing ones
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # Replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # Append any new messages to the end
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]
```

**Agent with Interrupt Before Action:**
```python
class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        
        # Interrupt before action for human approval
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    # ... other methods same as before ...
```

**Create Agent with Human Approval:**
```python
memory = SqliteSaver.from_conn_string(":memory:")
model = ChatOpenAI(model="gpt-3.5-turbo")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)
```

**Agent Pauses for Approval:**
```python
messages = [HumanMessage(content="Whats the weather in SF?")]
thread = {"configurable": {"thread_id": "1"}}

# Agent processes query but pauses before executing tool
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

# Check state - agent is waiting at 'action' node
print(abot.graph.get_state(thread).next)
# Output: ('action',)
```

**Continue After Human Approval:**
```python
# Human approves - continue execution
for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)

# Output: 
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}}
# Back to the model!
# [Weather results and final answer]
```

**Interactive Approval Loop:**
```python
messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "2"}}

# Initial query
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

# Loop to ask for approval at each step
while abot.graph.get_state(thread).next:
    print("\n", abot.graph.get_state(thread), "\n")
    _input = input("Proceed? (y/n): ")
    
    if _input != "y":
        print("Aborting")
        break
    
    # Continue execution
    for event in abot.graph.stream(None, thread):
        for v in event.values():
            print(v)
```

**Inspect Agent State:**
```python
# Get current state
state = abot.graph.get_state(thread)

print("Next nodes:", state.next)
print("Messages:", len(state.values['messages']))
print("Last message:", state.values['messages'][-1])
```

**Modify State Before Continuing:**
```python
# Get current state
state = abot.graph.get_state(thread)

# Modify the tool call before execution
tool_call = state.values['messages'][-1].tool_calls[0]
tool_call['args']['query'] = "weather in Los Angeles tomorrow"

# Update state with modified message
abot.graph.update_state(thread, {"messages": [state.values['messages'][-1]]})

# Continue with modified state
for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)
```

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
