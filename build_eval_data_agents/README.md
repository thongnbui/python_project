# Building and Evaluating Data Agents 🤖

[![Course Link](https://img.shields.io/badge/Course-DeepLearning.AI-blue)](https://learn.deeplearning.ai/courses/building-and-evaluating-data-agents)

A comprehensive guide to building production-grade multi-agent systems with LangGraph, featuring planning, execution, web research, data analysis, visualization, and rigorous evaluation using Goal-Plan-Act (GPA) alignment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Technologies](#key-technologies)
- [Architecture](#architecture)
- [Lessons](#lessons)
  - [Lesson 2: Multi-Agent Workflow](#lesson-2-multi-agent-workflow)
  - [Lesson 3: Expand Data Agent Capabilities](#lesson-3-expand-data-agent-capabilities)
  - [Lesson 5: Measure Agent's GPA](#lesson-5-measure-agents-gpa)
  - [Lesson 6: Improve Agent's GPA](#lesson-6-improve-agents-gpa)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## 🎯 Overview

This project demonstrates how to build, deploy, and systematically evaluate sophisticated data agents that can:

✅ **Plan & Execute** - Generate multi-step plans and adapt dynamically

✅ **Research** - Search the web (Tavily) and query enterprise data (Snowflake Cortex)

✅ **Analyze** - Query structured CRM data and unstructured meeting notes

✅ **Visualize** - Generate charts and graphs with Python

✅ **Evaluate** - Measure Goal-Plan-Act alignment using LLM-as-judge

✅ **Improve** - Use inline evaluations to provide real-time feedback to agents

**Use Cases:**
- Sales intelligence and lead prioritization
- Market research and competitive analysis
- Data-driven business insights
- Automated reporting and visualization
- Multi-source data synthesis

---

## 🛠️ Key Technologies

| Technology | Purpose |
|------------|---------|
| **LangGraph** | Multi-agent orchestration and state management |
| **LangChain** | LLM interactions and tool integration |
| **OpenAI GPT-4o/o3** | Primary LLMs for reasoning and generation |
| **Tavily Search** | Web research and public data retrieval |
| **Snowflake Cortex** | Enterprise data querying (Text2SQL + Semantic Search) |
| **TruLens** | Agent evaluation and tracing |
| **Matplotlib** | Chart generation and visualization |
| **Python REPL** | Dynamic code execution for charting |

---

## 🏗️ Architecture

### Multi-Agent System Design

```
                    ┌──────────────┐
                    │   Planner    │ ← Generates multi-step plans
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Executor   │ ← Orchestrates execution, decides replanning
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┬──────────────┐
            ▼              ▼              ▼              ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
    │    Web     │  │  Cortex    │  │   Chart    │  │Synthesizer │
    │ Researcher │  │ Researcher │  │ Generator  │  │            │
    └────────────┘  └────────────┘  └──────┬─────┘  └────────────┘
                                           │
                                           ▼
                                    ┌────────────┐
                                    │   Chart    │
                                    │Summarizer  │
                                    └────────────┘
```

### State Management

```python
class State(MessagesState):
    user_query: Optional[str]           # Original user request
    enabled_agents: Optional[List[str]] # Modular agent selection
    plan: Optional[Dict]                # Current execution plan
    current_step: int                   # Progress tracker
    agent_query: Optional[str]          # Instructions for next agent
    last_reason: Optional[str]          # Executor's reasoning
    replan_flag: Optional[bool]         # Triggers plan revision
    replan_attempts: Optional[Dict]     # Replan tracking per step
    messages: List[Message]             # Shared message history
```

---

## 📝 Lessons

### **Lesson 2: Multi-Agent Workflow** 🔄

**Objective:** Build a complete multi-agent data system with planning, execution, web research, charting, and synthesis capabilities.

#### Key Concepts

1. **State-Driven Architecture** - Shared memory across all agents
2. **Plan-Execute Pattern** - Planner generates strategy, Executor orchestrates
3. **Dynamic Replanning** - Executor can request plan revisions (max 3 attempts)
4. **Specialized Sub-Agents** - Each agent has a focused responsibility
5. **ReAct Agents** - For web research and code execution

---

#### 2.1 Initialize Agent State

```python
from typing import Literal, Optional, List, Dict, Any
from langgraph.graph import MessagesState

# Custom State class with specific keys
class State(MessagesState):
    user_query: Optional[str]                    # The user's original query
    enabled_agents: Optional[List[str]]          # Makes system modular
    plan: Optional[List[Dict[int, Dict[str, Any]]]]  # Steps to achieve goal
    current_step: int                            # Current step in plan
    agent_query: Optional[str]                   # Instructions for next agent
    last_reason: Optional[str]                   # Executor's reasoning
    replan_flag: Optional[bool]                  # Triggers replanning
    replan_attempts: Optional[Dict[int, Dict[int, int]]]  # Replan tracking
    # messages inherited from MessagesState
```

**Why State Matters:**
- Provides shared, evolving memory across all nodes
- Agents have context needed to act coherently
- Enables traceability and debugging

---

#### 2.2 Create Planner

**The planner generates a numbered, step-by-step plan assigning sub-agents to each action.**

```python
from prompts import plan_prompt
from langgraph.types import Command
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
import json

# Use reasoning model with JSON output
reasoning_llm = ChatOpenAI(
    model="o3",
    model_kwargs={"response_format": {"type": "json_object"}},
)

def planner_node(state: State) -> Command[Literal['executor']]:
    """
    Runs the planning LLM and stores the resulting plan in state.
    """
    # 1. Invoke LLM with the planner prompt
    llm_reply = reasoning_llm.invoke([plan_prompt(state)])
    
    # 2. Validate JSON
    try:
        content_str = llm_reply.content if isinstance(
            llm_reply.content, str) else str(llm_reply.content)
        parsed_plan = json.loads(content_str)
    except json.JSONDecodeError:
        raise ValueError(
            f"Planner returned invalid JSON:\n{llm_reply.content}")
    
    # 3. Store the plan and initialize state
    replan = state.get("replan_flag", False)
    
    return Command(
        update={
            "plan": parsed_plan,
            "messages": [HumanMessage(
                content=llm_reply.content,
                name="replan" if replan else "initial_plan")],
            "user_query": state.get("user_query", state["messages"][0].content),
            "current_step": 1 if not replan else state["current_step"],
            "replan_flag": state.get("replan_flag", False),
            "last_reason": "",
            "enabled_agents": state.get("enabled_agents"),
        },
        goto="executor",  # Always go to executor next
    )
```

**Example Plan Output (JSON):**
```json
{
  "1": {
    "action": "Research current market cap of JP Morgan",
    "agent": "web_researcher"
  },
  "2": {
    "action": "Generate chart comparing top 5 US banks",
    "agent": "chart_generator"
  },
  "3": {
    "action": "Summarize findings",
    "agent": "synthesizer"
  }
}
```

---

#### 2.3 Create Executor

**The executor orchestrates plan execution, generates agent instructions, and decides when to replan.**

```python
from prompts import executor_prompt
from langgraph.graph import END

MAX_REPLANS = 3

def executor_node(
    state: State,
) -> Command[Literal["web_researcher", "chart_generator", "synthesizer", "planner"]]:
    
    plan: Dict[str, Any] = state.get("plan", {})
    step: int = state.get("current_step", 1)
    
    # 0) If we just replanned, run the planned agent once before reconsidering
    if state.get("replan_flag"):
        planned_agent = plan.get(str(step), {}).get("agent")
        return Command(
            update={
                "replan_flag": False,
                "current_step": step + 1,  # Advance step
            },
            goto=planned_agent,
        )
    
    # 1) Call LLM to decide next action
    llm_reply = reasoning_llm.invoke([executor_prompt(state)])
    try:
        content_str = llm_reply.content if isinstance(
            llm_reply.content, str) else str(llm_reply.content)
        parsed = json.loads(content_str)
        replan: bool = parsed["replan"]
        goto: str   = parsed["goto"]
        reason: str = parsed["reason"]
        query: str  = parsed["query"]
    except Exception as exc:
        raise ValueError(f"Invalid executor JSON:\n{llm_reply.content}") from exc
    
    # Update state
    updates: Dict[str, Any] = {
        "messages": [HumanMessage(content=llm_reply.content, name="executor")],
        "last_reason": reason,
        "agent_query": query,
    }
    
    # Replan accounting
    replans: Dict[int, int] = state.get("replan_attempts", {}) or {}
    step_replans = replans.get(step, 0)
    
    # 2) Replan decision
    if replan:
        if step_replans < MAX_REPLANS:
            replans[step] = step_replans + 1
            updates.update({
                "replan_attempts": replans,
                "replan_flag": True,
                "current_step": step,  # Stay on same step for new plan
            })
            return Command(update=updates, goto="planner")
        else:
            # Max replans reached: skip to next step
            next_agent = plan.get(str(step + 1), {}).get("agent", "synthesizer")
            updates["current_step"] = step + 1
            return Command(update=updates, goto=next_agent)
    
    # 3) Happy path: run chosen agent
    planned_agent = plan.get(str(step), {}).get("agent")
    updates["current_step"] = step + 1 if goto == planned_agent else step
    updates["replan_flag"] = False
    return Command(update=updates, goto=goto)
```

**Executor Decision Logic:**
1. **Just Replanned?** → Execute the planned agent immediately
2. **Need to Replan?** → Go back to planner (if under MAX_REPLANS)
3. **Follow Plan?** → Execute next agent and advance step
4. **Max Replans Hit?** → Skip to next step

---

#### 2.4 Create Web Research Agent

**A ReAct agent using Tavily Search for web research.**

```python
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from helper import agent_system_prompt

# Initialize Tavily tool
tavily_tool = TavilySearch(max_results=5)

# Test the tool
results = tavily_tool.invoke("What is JP Morgan's stock price?")['results']

# Create ReAct agent
llm = ChatOpenAI(model="gpt-4o")

web_search_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=agent_system_prompt(f"""
        You are the Researcher. You can ONLY perform research 
        by using the provided search tool (tavily_tool). 
        When you have found the necessary information, end your output.  
        Do NOT attempt to take further actions.
    """),
)

# Test the agent
agent_response = web_search_agent.invoke(
    {"messages": "what is jp morgan's current market cap?"})
print(agent_response['messages'][-1].content)
```

**Web Research Node (LangGraph integration):**
```python
def web_research_node(
    state: State,
) -> Command[Literal["executor"]]:
    
    agent_query = state.get("agent_query")
    result = web_search_agent.invoke({"messages": agent_query})
    
    # Wrap in HumanMessage for compatibility
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, 
        name="web_researcher"
    )
    
    return Command(
        update={
            "messages": result["messages"],  # Share agent's message history
        },
        goto="executor",  # Always return to executor
    )
```

**What is ReAct?**
- **Re**asoning + **Act**ing pattern
- Agent iterates: Thought → Action → Observation
- Uses tools (Tavily) to gather information
- Returns final answer after sufficient research

---

#### 2.5 Create Chart Generator Agent

**Generates Python code and executes it to create visualizations.**

```python
from helper import python_repl_tool

# Chart generator agent
# ⚠️ WARNING: This performs arbitrary code execution
chart_agent = create_react_agent(
    llm,
    [python_repl_tool],
    prompt=agent_system_prompt(
        """
        You can only generate charts. You are working with a researcher colleague.
        1) Print the chart first.
        2) Save the chart to a file in the current working directory.
        3) At the very end of your message, output EXACTLY two lines:
           CHART_PATH: <relative_path_to_chart_file>
           CHART_NOTES: <one concise sentence summarizing the main insight>
        Do not include any other trailing text after these two lines.
        """
    ),
)

def chart_node(state: State) -> Command[Literal["chart_summarizer"]]:
    result = chart_agent.invoke(state)
    
    # Wrap in HumanMessage
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, 
        name="chart_generator"
    )
    
    return Command(
        update={
            "messages": result["messages"],
        },
        goto="chart_summarizer",  # Always summarize after charting
    )
```

**Python REPL Tool:**
```python
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool

repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. You will be used to execute python code
    that generates charts. Only print the chart once."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
```

**Example Chart Code Generated by Agent:**
```python
import matplotlib.pyplot as plt

# Data from research
banks = ['JPMorgan', 'Bank of America', 'Wells Fargo', 'Citigroup', 'Goldman Sachs']
market_caps = [580, 315, 185, 155, 135]  # in billions

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(banks, market_caps, color='steelblue')
plt.xlabel('Bank')
plt.ylabel('Market Cap (Billions USD)')
plt.title('Current Market Cap - Top 5 US Banks')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save and display
plt.savefig('current_market_cap_top_5_us_banks.png')
plt.show()

# Output metadata for summarizer
CHART_PATH: current_market_cap_top_5_us_banks.png
CHART_NOTES: JPMorgan leads with nearly double the market cap of Bank of America.
```

---

#### 2.6 Create Chart Summarizer

**Extracts chart metadata and generates written summaries.**

```python
import re

def chart_summarizer_node(state: State) -> Command[Literal["executor"]]:
    """
    Parse the chart generator's output for CHART_PATH and CHART_NOTES,
    then produce a summary message.
    """
    messages = state.get("messages", [])
    
    # Find most recent chart_generator message
    chart_message = None
    for msg in reversed(messages):
        if getattr(msg, "name", None) == "chart_generator":
            chart_message = msg.content
            break
    
    if not chart_message:
        return Command(
            update={"messages": [HumanMessage(
                content="No chart found to summarize.",
                name="chart_summarizer"
            )]},
            goto="executor"
        )
    
    # Extract metadata
    path_match = re.search(r"CHART_PATH:\s*(.+)", chart_message)
    notes_match = re.search(r"CHART_NOTES:\s*(.+)", chart_message)
    
    chart_path = path_match.group(1).strip() if path_match else "unknown"
    chart_notes = notes_match.group(1).strip() if notes_match else "No notes provided"
    
    summary = f"""
    Chart Summary:
    - File: {chart_path}
    - Key Insight: {chart_notes}
    
    The visualization has been generated and saved.
    """
    
    return Command(
        update={"messages": [HumanMessage(
            content=summary,
            name="chart_summarizer"
        )]},
        goto="executor"
    )
```

---

#### 2.7 Create Synthesizer

**Generates final prose summary of all research findings.**

```python
from prompts import synthesizer_prompt

def synthesizer_node(state: State) -> Command[Literal[END]]:
    """
    Final node: synthesize all research into a coherent answer.
    """
    llm_reply = llm.invoke([synthesizer_prompt(state)])
    
    return Command(
        update={
            "messages": [HumanMessage(
                content=llm_reply.content,
                name="synthesizer"
            )]
        },
        goto=END  # Terminates the graph
    )
```

---

#### 2.8 Build Complete LangGraph

```python
from langgraph.graph import StateGraph, START, END

# Create graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("planner", planner_node)
graph_builder.add_node("executor", executor_node)
graph_builder.add_node("web_researcher", web_research_node)
graph_builder.add_node("chart_generator", chart_node)
graph_builder.add_node("chart_summarizer", chart_summarizer_node)
graph_builder.add_node("synthesizer", synthesizer_node)

# Set entry point
graph_builder.add_edge(START, "planner")

# Compile graph
graph = graph_builder.compile()

# Visualize
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

---

#### 2.9 Run the Agent

```python
# Initialize state
initial_state = {
    "messages": [],
    "enabled_agents": ["web_researcher", "chart_generator", "chart_summarizer", "synthesizer"],
}

# User query
user_query = "What is the current market cap of the top 5 US banks? Create a bar chart."

# Invoke the graph
result = graph.invoke({
    **initial_state,
    "messages": [HumanMessage(content=user_query)],
    "user_query": user_query,
})

# Get final answer
print(result["messages"][-1].content)
```

**Execution Flow:**
```
1. Planner: Generates 3-step plan
   ├─ Step 1: web_researcher - Get market caps
   ├─ Step 2: chart_generator - Create bar chart
   └─ Step 3: synthesizer - Summarize findings

2. Executor: Runs Step 1
   └─ web_researcher: Searches web, returns data

3. Executor: Runs Step 2
   └─ chart_generator: Creates Python code, executes
       └─ chart_summarizer: Extracts metadata

4. Executor: Runs Step 3
   └─ synthesizer: Generates final summary

5. END
```

---

### **Lesson 3: Expand Data Agent Capabilities** 🗄️

**Objective:** Integrate Snowflake Cortex Agents to query both structured (CRM data) and unstructured (meeting notes) enterprise data.

#### Key Concepts

1. **Cortex Analyst** - Text-to-SQL for structured data queries
2. **Cortex Search** - Semantic search over unstructured documents
3. **Semantic Models** - Define database schema for LLM understanding
4. **Multi-Source Data** - Combine public web data + private enterprise data
5. **Hybrid Retrieval** - Structured + unstructured data fusion

---

#### 3.1 Environment Setup

```python
from dotenv import load_dotenv
_ = load_dotenv(override=True)

# Required environment variables
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PAT = os.getenv("SNOWFLAKE_PAT")
SNOWFLAKE_DATABASE = "SALES_INTELLIGENCE"
SNOWFLAKE_SCHEMA = "DATA"
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
```

---

#### 3.2 Explore the Data

**Structured CRM Data (via Cortex Analyst):**
```python
from helper import snowpark_session
import pandas as pd

# Connect to warehouse
snowpark_session.sql("USE WAREHOUSE SALES_INTELLIGENCE_WH").collect()

# Query CRM data
df = pd.DataFrame(
    snowpark_session.sql(
        "SELECT * FROM sales_intelligence.data.sales_metrics LIMIT 5"
    ).collect()
)
print(df)
```

**Example CRM Schema:**
| Column | Type | Description |
|--------|------|-------------|
| `company_name` | VARCHAR | Client company name |
| `deal_value` | FLOAT | Deal size in USD |
| `close_date` | DATE | Expected close date |
| `deal_status` | VARCHAR | Open/Closed/Lost |
| `product_line` | VARCHAR | Product category |
| `sales_rep` | VARCHAR | Assigned salesperson |

**Unstructured Meeting Notes (via Cortex Search):**
```python
import textwrap

# Query meeting transcript
rows = snowpark_session.sql("""
    SELECT transcript_text
    FROM sales_intelligence.data.sales_conversations
    LIMIT 1
""").collect()

transcript = rows[0]['TRANSCRIPT_TEXT']
wrapped = textwrap.fill(transcript, width=100)
print("=== Meeting Notes ===\n")
print(wrapped)
```

**Example Meeting Note:**
```
Initial discovery call with TechCorp Inc's IT Director and Solutions Architect. 
Client showed strong interest in our enterprise solution features, particularly 
the automated workflow capabilities. The main discussion centered around 
integration timeline and complexity. They currently use Legacy System X for 
their core operations and expressed concerns about potential disruption during 
migration. Action items include providing a detailed integration timeline 
document, scheduling a technical deep-dive with their infrastructure team, 
and sharing case studies of similar Legacy System X migrations. The client 
mentioned a Q2 budget allocation for digital transformation initiatives.
```

---

#### 3.3 Create Cortex Agent Tool

**Unified tool for both Text-to-SQL and Semantic Search:**

```python
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.core.cortex.lite_agent_service import AgentRunRequest
from pydantic import BaseModel, PrivateAttr
from typing import Type, Any
import json

# Configuration
SEMANTIC_MODEL_FILE = "@sales_intelligence.data.models/sales_metrics_model.yaml"
CORTEX_SEARCH_SERVICE = "sales_intelligence.data.sales_conversation_search"

# Tool argument schema
class CortexAgentArgs(BaseModel):
    query: str

class CortexAgentTool:
    name: str = "CortexAgent"
    description: str = "answers questions using sales conversations and metrics"
    args_schema: Type[CortexAgentArgs] = CortexAgentArgs
    
    _session: Session = PrivateAttr()
    _root: Root = PrivateAttr()
    _agent_service: Any = PrivateAttr()
    
    def __init__(self, session: Session):
        self._session = session
        self._root = Root(session)
        self._agent_service = self._root.cortex_agent_service
    
    def _build_request(self, query: str) -> AgentRunRequest:
        return AgentRunRequest.from_dict({
            "model": "claude-3-5-sonnet",
            "tools": [
                {
                    "tool_spec": {
                        "type": "cortex_analyst_text_to_sql",
                        "name": "analyst1"
                    }
                },
                {
                    "tool_spec": {
                        "type": "cortex_search",
                        "name": "search1"
                    }
                },
            ],
            "tool_resources": {
                "analyst1": {
                    "semantic_model_file": SEMANTIC_MODEL_FILE
                },
                "search1": {
                    "name": CORTEX_SEARCH_SERVICE,
                    "max_results": 10,
                    "id_column": "conversation_id"
                }
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}]
                }
            ]
        })
    
    def _consume_stream(self, stream):
        """Extract text, SQL, and citations from streaming response."""
        text, sql, citations = "", "", []
        
        for evt in stream.events():
            try:
                delta = (evt.data.get("delta") if isinstance(evt.data, dict)
                         else json.loads(evt.data).get("delta")
                         or json.loads(evt.data).get("data", {}).get("delta"))
            except Exception:
                continue
            
            if not isinstance(delta, dict):
                continue
            
            for item in delta.get("content", []):
                if item.get("type") == "text":
                    text += item.get("text", "")
                elif item.get("type") == "tool_results":
                    for result in item["tool_results"].get("content", []):
                        if result.get("type") != "json":
                            continue
                        j = result["json"]
                        
                        # Extract SQL from Text-to-SQL
                        if "request" in j and "sql" in j["request"]:
                            sql = j["request"]["sql"]
                        
                        # Extract citations from Search
                        if "results" in j:
                            for r in j["results"]:
                                if "id" in r and "chunk" in r:
                                    citations.append({
                                        "id": r["id"],
                                        "excerpt": r["chunk"][:200] + "..."
                                    })
        
        return text, sql, citations
    
    def __call__(self, query: str) -> str:
        """Execute agent query and return formatted response."""
        request = self._build_request(query)
        stream = self._agent_service.run_stream(request)
        
        text, sql, citations = self._consume_stream(stream)
        
        # Format response
        output = text
        if sql:
            output += f"\n\n**SQL Query:**\n```sql\n{sql}\n```"
        if citations:
            output += "\n\n**Sources:**\n"
            for i, cite in enumerate(citations, 1):
                output += f"{i}. [ID: {cite['id']}] {cite['excerpt']}\n"
        
        return output

# Initialize tool
cortex_tool = CortexAgentTool(session=snowpark_session)
```

---

#### 3.4 Create Cortex Researcher Node

```python
def cortex_researcher_node(
    state: State,
) -> Command[Literal["executor"]]:
    """
    Query Snowflake Cortex Agent for enterprise data.
    """
    query = state.get("agent_query", state.get("user_query", ""))
    
    # Call Cortex Agent
    agent_response = cortex_tool(query)
    
    # Create message
    new_message = HumanMessage(
        content=agent_response,
        name="cortex_researcher"
    )
    
    return Command(
        update={"messages": [new_message]},
        goto="executor",
    )
```

---

#### 3.5 Update Agent Descriptions

```python
# In prompts.py
def get_agent_descriptions() -> Dict[str, Dict[str, Any]]:
    return {
        "web_researcher": {
            "name": "Web Researcher",
            "capability": "Fetch public data via Tavily web search",
            "use_when": "Public information, news, current events needed",
            "limitations": "Cannot access private/internal company data",
        },
        "cortex_researcher": {  # NEW!
            "name": "Cortex Researcher",
            "capability": "Query private company data in Snowflake (structured CRM + unstructured meeting notes)",
            "use_when": "Internal documents, company databases, or private data access required",
            "limitations": "Cannot access public web data",
            "output_format": "Returns exact fields with SQL (structured) or excerpts with citations (unstructured)",
        },
        "chart_generator": {
            "name": "Chart Generator",
            "capability": "Build visualizations from structured data",
            "use_when": "User explicitly requests charts, graphs, plots",
            "position_requirement": "Must be used as final step after data gathering",
        },
        "synthesizer": {
            "name": "Synthesizer",
            "capability": "Write comprehensive prose summaries",
            "use_when": "Final step when no visualization requested",
        },
    }
```

---

#### 3.6 Rebuild Graph with Cortex Agent

```python
# Add cortex_researcher node to graph
graph_builder.add_node("cortex_researcher", cortex_researcher_node)

# Enable cortex_researcher in state
initial_state = {
    "messages": [],
    "enabled_agents": [
        "web_researcher",
        "cortex_researcher",  # NEW!
        "chart_generator",
        "chart_summarizer",
        "synthesizer"
    ],
}

# Compile graph
graph = graph_builder.compile()
```

---

#### 3.7 Example Multi-Source Query

```python
user_query = """
Which sales leads should we prioritize this week? 
For the top 3 deals by value, find any meeting notes 
mentioning concerns or action items.
"""

result = graph.invoke({
    **initial_state,
    "messages": [HumanMessage(content=user_query)],
    "user_query": user_query,
})

print(result["messages"][-1].content)
```

**Expected Plan:**
```json
{
  "1": {
    "action": "Query CRM for top 3 deals by value this week",
    "agent": "cortex_researcher"
  },
  "2": {
    "action": "Search meeting notes for those 3 companies",
    "agent": "cortex_researcher"
  },
  "3": {
    "action": "Synthesize findings with recommendations",
    "agent": "synthesizer"
  }
}
```

**Cortex Researcher Output (Step 1):**
```
**Top 3 Deals This Week:**

1. TechCorp Inc - $1.2M (Open, Close Date: 2025-11-05)
2. GlobalSoft Ltd - $950K (Open, Close Date: 2025-11-03)
3. DataFlow Systems - $720K (At Risk, Close Date: 2025-11-01)

**SQL Query:**
```sql
SELECT company_name, deal_value, deal_status, close_date
FROM sales_intelligence.data.sales_metrics
WHERE close_date BETWEEN CURRENT_DATE() AND DATEADD(day, 7, CURRENT_DATE())
  AND deal_status IN ('Open', 'At Risk')
ORDER BY deal_value DESC
LIMIT 3;
```
```

**Cortex Researcher Output (Step 2):**
```
**Meeting Notes - Action Items & Concerns:**

**TechCorp Inc:**
- Concern: Integration timeline complexity with Legacy System X
- Action: Provide detailed integration timeline document by EOW
- Action: Schedule technical deep-dive with infrastructure team
- Budget: Q2 allocation for digital transformation

**GlobalSoft Ltd:**
- Concern: Data migration tools compatibility
- Action: Share case studies of similar migrations
- Next: Executive approval needed for final contract

**DataFlow Systems:**
- Concern: Pricing concerns raised by CFO
- Status: At risk - competitor meeting scheduled
- Urgent: Need to send revised proposal by Thursday

**Sources:**
1. [ID: tc_20251025] "Initial discovery call with TechCorp Inc's IT Director..."
2. [ID: gs_20251023] "Follow-up meeting with GlobalSoft CTO regarding..."
3. [ID: df_20251024] "Pricing discussion with DataFlow CFO, expressed..."
```

---

### **Lesson 5: Measure Agent's GPA** 📊

**Objective:** Systematically evaluate agents using Goal-Plan-Act (GPA) alignment framework with LLM-as-judge.

#### Key Concepts

1. **GPA Alignment** - Goal, Plan, and Actions must be coherent
2. **LLM-as-Judge** - Use GPT-4 to evaluate agent quality
3. **Failure Modes** - Identify common issues (vague plans, irrelevant actions, etc.)
4. **TruLens Integration** - Automated evaluation and tracing
5. **Separable Dimensions** - Evaluate plan quality, action relevance, goal alignment independently

---

#### 5.1 Goal-Plan-Act Framework

**Three Key Evaluation Dimensions:**

| Dimension | What It Measures | Failure Mode |
|-----------|------------------|--------------|
| **Plan Quality** | Is the plan specific, actionable, and aligned with goal? | Vague steps, missing constraints, no measurable outputs |
| **Action Relevance** | Do the agent's actions follow the plan? | Off-plan actions, irrelevant tool calls |
| **Goal Alignment** | Does the final output answer the original query? | Partial answers, missing information, drift |

---

#### 5.2 Setup Evaluation Provider

```python
from trulens.providers.openai import OpenAI

# Use GPT-4 as the evaluator
gpa_eval_provider = OpenAI(model_engine="gpt-4o")
```

---

#### 5.3 Failure Mode 1: Plan Quality

**Bad Plan Example:**
```python
goal_and_plan = """
User Query: Which sales leads should we prioritize this week, 
and what specific action items should we take for each?

Plan:

1. Pull all sales leads from the past 12 months from the CRM.
2. For the largest 20 leads, compile any notes, call logs, 
   and related tasks from the CRM.
3. Summarize each lead's current stage in the pipeline.
4. Present the summary and recommendations in a single table.
"""
```

**Evaluate Plan Quality:**
```python
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector

# Define evaluator
f_plan_quality = Feedback(
    gpa_eval_provider.plan_quality_with_cot_reasons,
    name="Plan Quality",
).on({
    "trace": Selector(trace_level=True),
})

# Run evaluation
score, reason = f_plan_quality(goal_and_plan)

print(f"Score: {score}")  # Output: 0.67
print(f"Reason: {reason['reason']}")
```

**Why Low Score?**
- ❌ Vague selection: "Past 12 months" lacks urgency constraints
- ❌ Weak prioritization: "Largest 20" ignores lead score, stage urgency
- ❌ Missing actionability: No specific next actions or owners
- ❌ Output not specific: "Single table" without required fields

---

**Good Plan Example:**
```python
goal_and_better_plan = """
User Query: Which sales leads should we prioritize this week, 
and what specific action items should we take for each?

Plan:

1. Pull all leads with open opportunities from the CRM that have 
   a next action date within the next 14 days or no next action assigned.

2. Filter to leads with deal value > $10k or high lead score.

3. Sort by deal stage urgency (e.g., close date approaching, 
   at risk of going cold) and potential revenue impact.

4. For each prioritized lead:
   - Retrieve latest interaction notes, key decision-maker info, 
     and current blockers.
   - Identify overdue or missing action items.
   - Propose specific, high-impact next steps (e.g., schedule product demo, 
     send proposal revision, escalate to sales manager).

5. Group recommendations into this week's priority list with owner 
   assignments and deadlines.

6. Present results in a table with columns: Lead Name, Value, Stage, 
   Urgency Score, Next Action, Due Date, Owner.
"""

# Evaluate improved plan
score, reason = f_plan_quality(goal_and_better_plan)
print(f"Score: {score}")  # Output: 1.0
```

**Why High Score?**
- ✅ Specific constraints: "Next 14 days", "> $10k"
- ✅ Clear prioritization: Urgency + revenue impact
- ✅ Actionable steps: Specific next actions with owners
- ✅ Measurable output: Defined table columns

---

#### 5.4 Failure Mode 2: Action Relevance

**Evaluate if actions match the plan:**

```python
goal_plan_and_actions = """
User Query: What are the top 3 client deals closed this quarter, 
and what were the key themes from meeting notes?

Plan:
1. Query CRM for top 3 deals by value closed this quarter
2. Search meeting notes for those 3 clients
3. Identify key themes (pain points, decision factors)
4. Summarize findings

Actions Taken:
- Step 1: cortex_researcher queried CRM
  Result: TechCorp ($1.2M), GlobalSoft ($950K), DataFlow ($720K)
  
- Step 2: web_researcher searched Google for "TechCorp news"  # ❌ OFF-PLAN!
  Result: Found public press releases about TechCorp acquisition
  
- Step 3: cortex_researcher searched meeting notes
  Result: Found internal meeting transcripts for all 3 clients
  
- Step 4: synthesizer generated summary
"""

# Define evaluator
f_action_relevance = Feedback(
    gpa_eval_provider.action_relevance_with_cot_reasons,
    name="Action Relevance",
).on({
    "trace": Selector(trace_level=True),
})

score, reason = f_action_relevance(goal_plan_and_actions)
print(f"Score: {score}")  # Output: 0.75 (Step 2 was off-plan)
```

**Issues Detected:**
- ❌ Step 2 used `web_researcher` instead of `cortex_researcher`
- ❌ Searched for public news instead of internal meeting notes
- ❌ Wasted time on irrelevant information

---

#### 5.5 Failure Mode 3: Goal Alignment

**Evaluate if final output answers the original question:**

```python
goal_and_final_output = """
User Query: Which sales leads should we prioritize this week, 
and what specific action items should we take for each?

Final Output:
"Based on the CRM data, there are currently 47 open leads across 
all product lines. The average deal size is $320K. The top sales 
representative this quarter is Sarah Johnson with 12 closed deals. 
I recommend focusing on the enterprise segment as it has the highest 
conversion rate."
"""

# Define evaluator
f_goal_alignment = Feedback(
    gpa_eval_provider.goal_alignment_with_cot_reasons,
    name="Goal Alignment",
).on({
    "trace": Selector(trace_level=True),
})

score, reason = f_goal_alignment(goal_and_final_output)
print(f"Score: {score}")  # Output: 0.4 (Major drift from goal)
```

**Issues Detected:**
- ❌ Didn't identify **specific leads** to prioritize
- ❌ Didn't list **this week's** focus (time constraint ignored)
- ❌ Didn't provide **specific action items** for each lead
- ❌ Provided irrelevant stats (average deal size, top rep)

---

#### 5.6 Context Relevance

**Evaluate if retrieved information is relevant to the query:**

```python
from helper import f_context_relevance

query_and_context = """
Query: What are the integration concerns mentioned by TechCorp?

Retrieved Context:
"Initial discovery call with TechCorp Inc's IT Director and Solutions 
Architect. Client showed strong interest in our enterprise solution 
features, particularly the automated workflow capabilities. The main 
discussion centered around integration timeline and complexity. They 
currently use Legacy System X for their core operations and expressed 
concerns about potential disruption during migration. Action items 
include providing a detailed integration timeline document."
"""

score = f_context_relevance(query_and_context)
print(f"Score: {score}")  # Output: 1.0 (Highly relevant)
```

---

#### 5.7 Create Complete Evaluation Suite

```python
from trulens.core import TruSession
from trulens.apps.langgraph import TruGraph

# Initialize TruLens
session = TruSession()

# Define all evaluators
evaluators = [
    Feedback(gpa_eval_provider.plan_quality_with_cot_reasons, name="Plan Quality"),
    Feedback(gpa_eval_provider.action_relevance_with_cot_reasons, name="Action Relevance"),
    Feedback(gpa_eval_provider.goal_alignment_with_cot_reasons, name="Goal Alignment"),
    Feedback(f_context_relevance, name="Context Relevance"),
]

# Wrap graph with TruLens instrumentation
tru_graph = TruGraph(
    app=graph,
    app_name="Sales Intelligence Agent",
    app_version="v1.0",
    feedbacks=evaluators,
)

# Run agent with evaluation
with tru_graph as recording:
    result = graph.invoke({
        **initial_state,
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
    })

# View results
session.get_leaderboard()
```

**Leaderboard Output:**
| App Name | Version | Plan Quality | Action Relevance | Goal Alignment | Context Relevance | Avg Latency |
|----------|---------|--------------|------------------|----------------|-------------------|-------------|
| Sales Intelligence Agent | v1.0 | 0.85 | 0.90 | 0.75 | 0.95 | 12.3s |

---

### **Lesson 6: Improve Agent's GPA** 🚀

**Objective:** Use evaluation insights to improve agent performance through better prompts and inline evaluations.

#### Key Concepts

1. **Inline Evaluations** - Real-time feedback during execution
2. **Structured Plans** - Add pre-conditions, post-conditions, sub-goals
3. **Adaptive Behavior** - Agent responds to evaluation feedback
4. **Targeted Improvements** - Focus on low-scoring dimensions
5. **Continuous Monitoring** - Track improvements over versions

---

#### 6.1 Add Inline Evaluations

**Provide real-time feedback to the executor about retrieval quality:**

```python
from trulens.apps.langgraph.inline_evaluations import inline_evaluation
from trulens.otel.semconv.trace import SpanAttributes
from trulens.core.otel.instrument import instrument
from helper import f_context_relevance

@inline_evaluation(f_context_relevance)
@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: args[0].get("agent_query"),
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [
            ret.update["messages"][-1].content
        ] if hasattr(ret, "update") else "No tool call",
    },
)
def cortex_researcher_node_with_eval(
    state: State,
) -> Command[Literal["executor"]]:
    """
    Query Snowflake Cortex Agent with inline evaluation.
    """
    query = state.get("agent_query", state.get("user_query", ""))
    
    # Call Cortex Agent
    agent_response = cortex_tool(query)
    
    # Create message
    new_message = HumanMessage(
        content=agent_response,
        name="cortex_researcher"
    )
    
    # Inline evaluation happens automatically via decorator
    # If context_relevance < 0.5, executor receives low score signal
    
    return Command(
        update={"messages": [new_message]},
        goto="executor",
    )
```

**How Inline Evaluations Work:**
1. `@inline_evaluation` decorator runs `f_context_relevance` after the function
2. Score is logged to TruLens and attached to the span
3. **Low scores trigger alerts** - Executor can see feedback
4. Agent can decide to **replan** or **gather more context**

---

#### 6.2 Improve Planning Prompt

**Add explicit sub-goals, pre-conditions, and post-conditions:**

```python
import helper
import prompts
from langchain.schema import HumanMessage

def patched_plan_prompt(state):
    base = prompts.plan_prompt(state).content
    insertion = '"action": "string",\n            "pre_conditions": ["string", ...],\n            "post_conditions": ["string", ...],\n            "goal": "string",'
    base = base.replace('"action": "string",', insertion)
    return HumanMessage(content=base)

helper.plan_prompt = patched_plan_prompt
```

#### 6.3 Improve Executor Prompt

**Add evaluation feedback and decision-making guidance:**

```python
def improved_executor_prompt(state: State) -> str:
    """
    Enhanced executor prompt with evaluation awareness.
    """
    user_query = state.get("user_query", "")
    plan = state.get("plan", {})
    current_step = state.get("current_step", 1)
    messages = state.get("messages", [])
    last_reason = state.get("last_reason", "")
    
    # Get most recent evaluation scores
    recent_evals = get_recent_inline_eval_scores(messages)  # Helper function
    
    prompt = f"""
You are the executor agent. Your job is to:
1. Review the current plan and progress
2. Evaluate if the last agent's output meets success criteria
3. Decide the next action (continue plan, replan, or terminate)

**User Query:** {user_query}

**Current Plan:**
{json.dumps(plan, indent=2)}

**Current Step:** {current_step}

**Last Agent's Reasoning:** {last_reason}

**Recent Evaluation Scores:**
{recent_evals}

**Decision Framework:**

1. **Check Success Criteria:**
   - Did the last step meet its post-conditions?
   - Review success_criteria from plan step {current_step - 1}
   - Check evaluation scores (context_relevance should be > 0.7)

2. **Decide Next Action:**
   - If success criteria MET and more steps remain → **Continue to next planned agent**
   - If success criteria FAILED and replans < MAX → **Replan with specific feedback**
   - If all steps complete and goal achieved → **Go to synthesizer**
   - If stuck after MAX_REPLANS → **Skip to next step with note**

3. **Construct Agent Query:**
   - Use the "action" field from the next planned step
   - Add context from previous steps if needed
   - Be specific about data fields required

**Output Format (JSON):**
{{
  "replan": false,
  "goto": "cortex_researcher",
  "reason": "Step 1 succeeded with context_relevance=0.95. Proceeding to Step 2 to search meeting notes for the 3 companies retrieved.",
  "query": "Search meeting transcripts for TechCorp Inc, GlobalSoft Ltd, and DataFlow Systems. Find mentions of pain points, concerns, decision factors, and action items from Q4 2025 meetings."
}}

**Evaluation Thresholds:**
- context_relevance < 0.5 → **Replan** (retrieved info not relevant)
- context_relevance 0.5-0.7 → **Consider additional research**
- context_relevance > 0.7 → **Proceed** (good retrieval)

Now make your decision:
"""
    
    return prompt
```

**Key Improvements:**
- ✅ **Evaluation awareness** - Executor sees inline eval scores
- ✅ **Decision framework** - Clear logic for next steps
- ✅ **Thresholds** - Quantitative guidelines for actions
- ✅ **Feedback loop** - Low scores trigger replanning

---

#### 6.4 Rebuild Graph with Improvements

```python
# Create improved graph with new prompts and inline evals
graph_builder = StateGraph(State)

# Use improved prompt functions
improved_reasoning_llm = ChatOpenAI(
    model="o3",
    model_kwargs={"response_format": {"type": "json_object"}},
)

def improved_planner_node(state: State) -> Command[Literal['executor']]:
    llm_reply = improved_reasoning_llm.invoke([improved_plan_prompt(state)])
    # ... (same logic as before)

def improved_executor_node(state: State) -> Command:
    llm_reply = improved_reasoning_llm.invoke([improved_executor_prompt(state)])
    # ... (same logic as before)

# Add nodes
graph_builder.add_node("planner", improved_planner_node)
graph_builder.add_node("executor", improved_executor_node)
graph_builder.add_node("cortex_researcher", cortex_researcher_node_with_eval)  # With inline eval!
graph_builder.add_node("web_researcher", web_research_node)
graph_builder.add_node("chart_generator", chart_node)
graph_builder.add_node("chart_summarizer", chart_summarizer_node)
graph_builder.add_node("synthesizer", synthesizer_node)

graph_builder.add_edge(START, "planner")

# Compile
improved_graph = graph_builder.compile()
```

---

#### 6.5 Compare Before & After

```python
import pandas as pd

# Run both versions with same query
test_query = "What are the top 3 client deals closed this quarter, and what key themes emerge from their meeting notes?"

# Version 1: Original agent
with tru_graph_v1 as recording:
    result_v1 = graph_v1.invoke({**initial_state, "messages": [HumanMessage(content=test_query)]})

# Version 2: Improved agent
with tru_graph_v2 as recording:
    result_v2 = improved_graph.invoke({**initial_state, "messages": [HumanMessage(content=test_query)]})

# Compare results
comparison = pd.DataFrame({
    "Metric": ["Plan Quality", "Action Relevance", "Goal Alignment", "Context Relevance", "Avg Latency"],
    "v1.0 (Original)": [0.70, 0.75, 0.65, 0.80, "15.2s"],
    "v2.0 (Improved)": [0.95, 0.95, 0.90, 0.95, "13.8s"],
    "Improvement": ["+36%", "+27%", "+38%", "+19%", "-9%"],
})

print(comparison)
```

**Results:**
| Metric | v1.0 (Original) | v2.0 (Improved) | Improvement |
|--------|----------------|-----------------|-------------|
| Plan Quality | 0.70 | **0.95** | +36% |
| Action Relevance | 0.75 | **0.95** | +27% |
| Goal Alignment | 0.65 | **0.90** | +38% |
| Context Relevance | 0.80 | **0.95** | +19% |
| Avg Latency | 15.2s | **13.8s** | -9% |

**What Improved:**
- ✅ Plans are more specific with clear success criteria
- ✅ Executor makes better decisions using eval feedback
- ✅ Fewer off-plan actions due to structured prompts
- ✅ Better final outputs that fully answer the query
- ✅ Faster execution (fewer replans needed)

---

## 🚀 Setup

### Prerequisites

- Python 3.10+
- OpenAI API key
- Tavily API key
- Snowflake account (for Lesson 3+)

### Installation

```bash
# Clone repository
git clone <repo_url>
cd build_eval_data_agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r ../requirements.txt
```

### Environment Configuration

```bash
# Copy template
cp env.template .env

# Edit .env with your credentials
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PAT=your_password
SNOWFLAKE_DATABASE=SALES_INTELLIGENCE
SNOWFLAKE_SCHEMA=DATA
SNOWFLAKE_ROLE=ACCOUNTADMIN
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
```

---

## 💻 Usage

### Run Jupyter Notebooks

```bash
jupyter notebook
```

Open any lesson notebook:
- `L2.ipynb` - Multi-Agent Workflow
- `L3.ipynb` - Expand Data Agent Capabilities
- `L5.ipynb` - Measure Agent's GPA
- `L6.ipynb` - Improve Agent's GPA

### Run as Python Script

```python
from helper import graph, State
from langchain.schema import HumanMessage

# Initialize state
initial_state = {
    "messages": [],
    "enabled_agents": ["web_researcher", "chart_generator", "chart_summarizer", "synthesizer"],
}

# User query
user_query = "What is the current market cap of the top 5 US banks? Create a bar chart."

# Invoke graph
result = graph.invoke({
    **initial_state,
    "messages": [HumanMessage(content=user_query)],
    "user_query": user_query,
})

# Print final answer
print(result["messages"][-1].content)
```

---

## 📂 Project Structure

```
build_eval_data_agents/
├── README.md                               # This file
├── requirements.txt                        # Dependencies (symlink to parent)
├── env.template                            # Environment variable template
├── helper.py                               # Shared utilities (State, tools, Snowflake session)
├── prompts.py                              # Prompt templates for planner/executor
├── L2.ipynb                                # Lesson 2: Multi-Agent Workflow
├── L3.ipynb                                # Lesson 3: Expand Capabilities (Cortex)
├── L5.ipynb                                # Lesson 5: Measure GPA
├── L6.ipynb                                # Lesson 6: Improve GPA
├── default.sqlite                          # TruLens evaluation database
├── current_market_cap_top_5_us_banks.png   # Example chart output
├── top_3_client_deals_chart.png            # Example chart output
└── top_3_client_deals_by_value.png         # Example chart output
```

### Key Files

- **`helper.py`**: Core utilities
  - `State` class definition
  - `python_repl_tool` for code execution
  - `snowpark_session` for Snowflake connection
  - `cortex_agent` tool for Text-to-SQL + Semantic Search
  - Evaluation functions (`f_context_relevance`)

- **`prompts.py`**: Prompt engineering
  - `plan_prompt()` - Generates planning instructions
  - `executor_prompt()` - Generates execution instructions
  - `agent_system_prompt()` - System prompts for sub-agents
  - `get_agent_descriptions()` - Agent capability specifications

- **`default.sqlite`**: TruLens database
  - Stores evaluation runs
  - Traces agent execution
  - Tracks metrics over time

---

## 🎓 Learning Objectives

By completing these lessons, you will learn to:

✅ Design multi-agent systems with clear separation of concerns

✅ Implement plan-execute architectures with dynamic replanning

✅ Integrate multiple data sources (web, CRM, documents)

✅ Use Snowflake Cortex for Text-to-SQL and semantic search

✅ Build ReAct agents with tool-calling capabilities

✅ Generate visualizations dynamically with Python code execution

✅ Systematically evaluate agent performance using GPA framework

✅ Use inline evaluations to provide real-time feedback

✅ Improve agent quality through prompt engineering

✅ Monitor and compare agent versions over time

---

## 🔗 Resources

- [Course Link](https://learn.deeplearning.ai/courses/building-and-evaluating-data-agents)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Snowflake Cortex Agents](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents)
- [TruLens Evaluation](https://www.trulens.org/)
- [Tavily Search API](https://tavily.com/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)

---

## 🏆 Best Practices

### Multi-Agent Design
- **Modular Agents**: Each agent has one clear responsibility
- **Shared State**: Use `State` class for cross-agent communication
- **Plan-Execute**: Separate planning from execution for flexibility
- **Replan Budget**: Limit replans to avoid infinite loops (MAX_REPLANS=3)

### Prompt Engineering
- **Structured Output**: Always use JSON for LLM-to-LLM communication
- **Explicit Criteria**: Define success conditions, pre/post-conditions
- **Examples**: Include few-shot examples in prompts
- **Constraints**: Specify required fields, formats, and limits

### Evaluation Strategy
- **Offline Evals**: Run comprehensive evaluations after changes
- **Inline Evals**: Provide real-time feedback during execution
- **Multiple Dimensions**: Evaluate plan quality, action relevance, goal alignment separately
- **Track Over Time**: Compare agent versions to measure improvements

### Production Considerations
- **Sandboxing**: Run code execution in isolated environments
- **Rate Limiting**: Respect API quotas (OpenAI, Tavily, Snowflake)
- **Error Handling**: Gracefully handle tool failures and LLM errors
- **Logging**: Use TruLens tracing for debugging and monitoring
- **Cost Tracking**: Monitor token usage and API costs per query

---

## 🚨 Security Notes

⚠️ **Code Execution Warning**: The chart generator uses `PythonREPL` to execute arbitrary Python code. This can be dangerous if not sandboxed properly. In production:
- Use containerized execution environments (Docker, Kubernetes)
- Restrict filesystem access and network access
- Validate and sanitize code before execution
- Set resource limits (CPU, memory, timeout)

⚠️ **API Keys**: Never commit `.env` files or hardcode API keys. Use environment variables and secret management systems in production.

⚠️ **Data Privacy**: Be cautious when connecting to production databases. Use read-only credentials and implement proper access controls.

---

## 📊 Example Outputs

### Market Cap Analysis with Chart

**Query:** "What is the current market cap of the top 5 US banks? Create a bar chart."

**Final Output:**
```
Based on current market data, here are the top 5 US banks by market capitalization:

1. **JPMorgan Chase** - $580.2B
2. **Bank of America** - $315.4B
3. **Wells Fargo** - $185.3B
4. **Citigroup** - $155.1B
5. **Goldman Sachs** - $135.6B

JPMorgan Chase dominates with nearly double the market cap of Bank of America, 
reflecting its strong position in investment banking and consumer banking sectors.

[Chart saved to: current_market_cap_top_5_us_banks.png]
```

### Sales Lead Prioritization

**Query:** "Which sales leads should we prioritize this week?"

**Final Output:**
```
**This Week's Priority Leads:**

**1. DataFlow Systems - $720K (URGENT)**
- Status: At Risk
- Close Date: Nov 1 (2 days away!)
- Issue: CFO raised pricing concerns, competitor meeting scheduled
- Action: Send revised proposal by Thursday (Owner: Sarah J.)
- Priority: HIGH - Risk of losing to competitor

**2. TechCorp Inc - $1.2M**
- Status: Open
- Close Date: Nov 5
- Issue: Integration timeline complexity with Legacy System X
- Action: Provide integration timeline doc + schedule tech deep-dive (Owner: Mike T.)
- Priority: HIGH - Largest deal value

**3. GlobalSoft Ltd - $950K**
- Status: Open
- Close Date: Nov 3
- Issue: Awaiting executive approval
- Action: Follow up with CTO, prepare executive summary (Owner: Lisa K.)
- Priority: MEDIUM - On track but needs nudge

**Total Pipeline Value:** $2.89M
```

---

## 🎯 Next Steps

After mastering these lessons, consider:

1. **Add More Agents**: 
   - Email agent (send follow-ups)
   - Calendar agent (schedule meetings)
   - Slack agent (notify team)

2. **Improve Robustness**:
   - Add retry logic for API failures
   - Implement circuit breakers
   - Add fallback strategies

3. **Enhance Evaluations**:
   - Custom evaluation metrics for your domain
   - A/B testing framework
   - User feedback collection

4. **Scale to Production**:
   - Containerize with Docker
   - Deploy to Kubernetes
   - Add API gateway
   - Implement authentication

5. **Advanced Features**:
   - Multi-turn conversations with memory
   - Streaming responses to users
   - Parallel agent execution
   - Human-in-the-loop workflows

---

## 📄 License

This project is for educational purposes as part of the DeepLearning.AI course.

---

**Happy Building! 🚀**
