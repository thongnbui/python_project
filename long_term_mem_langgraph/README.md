# Long-Term Memory with LangGraph 📧

A comprehensive guide to building an intelligent email assistant with progressively advanced memory capabilities using LangGraph, demonstrating semantic, episodic, and procedural memory patterns.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Technologies](#key-technologies)
- [Architecture](#architecture)
- [Lessons](#lessons)
  - [Lesson 2: Baseline Email Assistant](#lesson-2-baseline-email-assistant-)
  - [Lesson 3: Email Assistant with Semantic Memory](#lesson-3-email-assistant-with-semantic-memory-)
  - [Lesson 4: Email Assistant with Semantic + Episodic Memory](#lesson-4-email-assistant-with-semantic--episodic-memory-)
  - [Lesson 5: Email Assistant with Semantic + Episodic + Procedural Memory](#lesson-5-email-assistant-with-semantic--episodic--procedural-memory-)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## 🎯 Overview

This project demonstrates how to build an intelligent email assistant that progressively learns and adapts through three types of memory:

✅ **Semantic Memory** - Remembers facts, details, and information from previous emails

✅ **Episodic Memory** - Learns from examples (few-shot learning) to improve classification

✅ **Procedural Memory** - Dynamically updates instructions and prompts based on user feedback

**Core Capabilities:**
- **Email Triage** - Classifies incoming emails (respond, ignore, notify)
- **Response Generation** - Drafts contextual email responses
- **Meeting Scheduling** - Schedules meetings and checks calendar availability
- **Memory Management** - Stores and retrieves relevant information across sessions
- **Human-in-the-Loop** - Allows user feedback to refine classification
- **Adaptive Instructions** - Updates behavior based on user preferences

**Use Cases:**
- Executive email assistants
- Personal productivity tools
- Customer support automation
- Intelligent email routing systems

---

## 🛠️ Key Technologies

| Technology | Purpose |
|------------|---------|
| **LangGraph** | Agent workflow orchestration and state management |
| **LangChain** | LLM interactions and tool integration |
| **LangMem** | Memory management tools (manage_memory, search_memory) |
| **InMemoryStore** | Vector store for semantic search and memory storage |
| **OpenAI GPT-4o-mini** | Primary LLM for classification and generation |
| **Pydantic** | Structured output validation |
| **Python-dotenv** | Environment variable management |

---

## 🏗️ Architecture

### Email Assistant Workflow

```
                    ┌──────────────┐
                    │ Email Input  │ ← Incoming email
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Triage     │ ← Classify: respond/ignore/notify
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  IGNORE  │   │  NOTIFY  │   │ RESPOND │
    └──────────┘   └──────────┘   └────┬─────┘
                                       │
                                       ▼
                              ┌──────────────┐
                              │Response Agent│ ← Drafts response, schedules meetings
                              └──────────────┘
```

### Memory Architecture (L3+)

```
┌─────────────────────────────────────────┐
│         InMemoryStore                   │
│  ┌─────────────────────────────────┐   │
│  │  Semantic Memory (embeddings)   │   │ ← Facts, details, contacts
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Episodic Memory (examples)     │   │ ← Few-shot examples (L4+)
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Procedural Memory (prompts)    │   │ ← Instructions, rules (L5+)
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### State Management

```python
class State(TypedDict):
    email_input: dict              # Incoming email data
    messages: Annotated[list, add_messages]  # Conversation history
```

---

## 📝 Lessons

### **Lesson 2: Baseline Email Assistant** 📧

**Objective:** Build a basic email assistant that classifies incoming emails and handles responses using hard-coded rules and tools.

#### Key Concepts

1. **Email Triage** - Classify emails into three categories (respond, ignore, notify)
2. **Structured Output** - Use Pydantic models for reliable classification
3. **ReAct Agent** - Tool-using agent for response generation
4. **LangGraph Workflow** - State-based routing between triage and response
5. **Tool Integration** - Email writing, meeting scheduling, calendar checking

---

#### 2.1 Setup Profile and Rules

```python
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}
```

---

#### 2.2 Define Triage Router

**Classification Model:**

```python
from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o-mini")

class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""
    
    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

llm_router = llm.with_structured_output(Router)
```

**Triage Prompt:**

```python
from prompts import triage_system_prompt, triage_user_prompt

# System prompt includes role, background, rules, and few-shot examples
# User prompt formats the email for classification
```

---

#### 2.3 Define Response Agent Tools

```python
from langchain_core.tools import tool

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"
```

---

#### 2.4 Create Response Agent

```python
from langgraph.prebuilt import create_react_agent
from prompts import agent_system_prompt, create_prompt

agent = create_react_agent(
    "openai:gpt-4o",
    tools=[write_email, schedule_meeting, check_calendar_availability],
    prompt=create_prompt,
)
```

---

#### 2.5 Build LangGraph Workflow

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    # Classify email and route accordingly
    result = llm_router.invoke([system_prompt, user_prompt])
    
    if result.classification == "respond":
        return Command(
            update={"messages": [{"role": "user", "content": f"Respond to {state['email_input']}"}]},
            goto="response_agent"
        )
    elif result.classification == "ignore":
        return Command(goto=END)
    else:  # notify
        return Command(goto=END)

workflow = StateGraph(State)
workflow.add_node("triage", triage_router)
workflow.add_node("response_agent", agent)
workflow.add_edge(START, "triage")
workflow.add_edge("response_agent", END)

graph = workflow.compile()
```

---

#### 2.6 Run the Assistant

```python
email = {
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "body": "Hi John, I was reviewing the API documentation..."
}

result = graph.invoke({
    "email_input": email,
    "messages": []
})
```

**What You'll Learn:**
- How to build a basic email triage system
- How to use structured output for reliable classification
- How to create a ReAct agent with tools
- How to build LangGraph workflows with conditional routing
- How to integrate multiple tools (email, calendar, scheduling)

**Limitations:**
- ❌ No memory - cannot remember previous emails or context
- ❌ Static rules - classification based only on hard-coded rules
- ❌ No learning - cannot improve from examples or feedback

**Next Steps:**
Lesson 3 adds semantic memory, enabling the assistant to remember details from previous emails and use that context for better responses.

---

### **Lesson 3: Email Assistant with Semantic Memory** 🧠

**Objective:** Add semantic memory capabilities to the email assistant, enabling it to remember details from previous emails and use that context for better classification and responses.

#### Key Differences from Lesson 2

| Feature | Lesson 2 | Lesson 3 |
|---------|----------|----------|
| **Memory** | ❌ None | ✅ Semantic memory (InMemoryStore) |
| **Context** | ❌ No previous email context | ✅ Can search and retrieve past emails |
| **Personalization** | ❌ Generic responses | ✅ Context-aware responses using stored information |
| **Tools** | 3 tools (email, calendar, schedule) | 5 tools (+ manage_memory, search_memory) |

---

#### Key Concepts

1. **Semantic Memory** - Store and retrieve information using embeddings
2. **InMemoryStore** - Vector store for semantic search
3. **Memory Tools** - `manage_memory` and `search_memory` from langmem
4. **Context Retrieval** - Search past emails and stored information
5. **Embedding-Based Search** - Find relevant information using semantic similarity

---

#### 3.1 Setup Memory Store

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)
```

**What is InMemoryStore?**
- Vector store that uses embeddings for semantic search
- Stores key-value pairs with optional metadata
- Supports namespacing for organizing different types of data
- Enables similarity search across stored content

---

#### 3.2 Create Memory Management Tools

```python
from langmem import create_manage_memory_tool, create_search_memory_tool

manage_memory_tool = create_manage_memory_tool(
    namespace=("email_assistant", "lance"),  # Namespace for organizing memories
    store=store,
)

search_memory_tool = create_search_memory_tool(
    namespace=("email_assistant", "lance"),
    store=store,
)
```

**Memory Tool Capabilities:**

**`manage_memory_tool`:**
- Store new information (contacts, preferences, facts)
- Update existing memories
- Delete outdated information

**`search_memory_tool`:**
- Search stored memories by semantic similarity
- Retrieve relevant context for current email
- Find related information from past interactions

---

#### 3.3 Update Response Agent with Memory

```python
agent_system_prompt_memory = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage {name}'s communications and schedule:

1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots for a given day
4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
5. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{instructions}
</ Instructions >
"""

agent = create_react_agent(
    "openai:gpt-4o",
    tools=[write_email, schedule_meeting, check_calendar_availability, 
           manage_memory_tool, search_memory_tool],
    prompt=create_prompt,
)
```

---

#### 3.4 Example: Using Memory in Responses

**Scenario:** Alice previously mentioned she prefers afternoon meetings.

```python
# Agent receives email from Alice requesting a meeting
# 1. Agent searches memory: search_memory("Alice meeting preferences")
# 2. Retrieves: "Alice prefers afternoon meetings"
# 3. Agent checks calendar and suggests afternoon slots
# 4. Agent stores new information: manage_memory("Alice meeting scheduled for Tuesday 2pm")
```

**Benefits:**
- ✅ Remembers contact preferences
- ✅ Recalls previous conversations
- ✅ Builds context over time
- ✅ Provides personalized responses

---

**What You'll Learn:**
- How to set up InMemoryStore for semantic memory
- How to use langmem tools for memory management
- How to integrate memory tools into ReAct agents
- How to search and retrieve relevant context
- How semantic embeddings enable similarity search

**Key Improvements:**
- ✅ **Context Awareness** - Can reference previous emails and stored information
- ✅ **Personalization** - Remembers preferences and details about contacts
- ✅ **Information Persistence** - Stores facts across multiple email interactions

**Limitations:**
- ❌ No few-shot learning - cannot learn from example classifications
- ❌ No user feedback loop - cannot refine behavior based on corrections
- ❌ Static prompts - instructions cannot be updated dynamically

**Next Steps:**
Lesson 4 adds episodic memory (few-shot examples) and human-in-the-loop feedback to improve classification accuracy.

---

### **Lesson 4: Email Assistant with Semantic + Episodic Memory** 🎯

**Objective:** Add episodic memory (few-shot examples) and human-in-the-loop feedback to refine the assistant's email classification accuracy.

#### Key Differences from Lesson 3

| Feature | Lesson 3 | Lesson 4 |
|---------|-----------|----------|
| **Memory Types** | ✅ Semantic only | ✅ Semantic + Episodic |
| **Learning** | ❌ No examples | ✅ Few-shot examples from memory |
| **User Feedback** | ❌ None | ✅ Human-in-the-loop after triage |
| **Classification** | Static rules + context | Dynamic examples + rules + context |
| **Triage Prompt** | Hard-coded examples | Examples retrieved from memory |

---

#### Key Concepts

1. **Episodic Memory** - Store specific examples (email + classification) for few-shot learning
2. **Human-in-the-Loop** - User can correct classification after triage step
3. **Few-Shot Learning** - Use stored examples to improve classification accuracy
4. **Example Retrieval** - Search and retrieve similar email examples from memory
5. **Dynamic Prompting** - Include retrieved examples in triage prompt

---

#### 4.1 Store Episodic Memory Examples

**Store Example Classifications:**

```python
import uuid
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)

# Example 1: Email that should be responded to
email_example_1 = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": "Hi John, I was reviewing the API documentation..."
}

data_1 = {
    "email": email_example_1,
    "label": "respond"  # Classification label
}

store.put(
    ("email_assistant", "lance", "examples"),  # Namespace: (app, user, type)
    str(uuid.uuid4()),  # Unique ID
    data_1
)

# Example 2: Email that should be ignored
email_example_2 = {
    "author": "Sarah Chen <sarah.chen@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Update: Backend API Changes Deployed to Staging",
    "email_thread": "Hi John, Just wanted to let you know..."
}

data_2 = {
    "email": email_example_2,
    "label": "ignore"
}

store.put(
    ("email_assistant", "lance", "examples"),
    str(uuid.uuid4()),
    data_2
)
```

**Episodic Memory Structure:**
- **Namespace**: `("email_assistant", "lance", "examples")` - Organizes examples by app, user, and type
- **Key**: Unique UUID for each example
- **Value**: Dictionary with email data and classification label

---

#### 4.2 Retrieve Examples for Few-Shot Learning

**Search Similar Emails:**

```python
from langmem import create_search_memory_tool

search_memory_tool = create_search_memory_tool(
    namespace=("email_assistant", "lance", "examples"),
    store=store,
)

# Search for similar emails
results = search_memory_tool.invoke({
    "query": "API documentation question",
    "limit": 3  # Retrieve top 3 similar examples
})
```

**Format Examples for Prompt:**

```python
template = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content: 
```
{content}
```
> Triage Result: {result}"""

def format_few_shot_examples(examples):
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            template.format(
                subject=eg.value["email"]["subject"],
                to_email=eg.value["email"]["to"],
                from_email=eg.value["email"]["author"],
                content=eg.value["email"]["email_thread"][:400],
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)
```

---

#### 4.3 Update Triage Prompt with Examples

```python
triage_system_prompt = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Background >
{user_profile_background}. 
</ Background >

< Instructions >

{name} gets lots of emails. Your job is to categorize each email into one of three categories:

1. IGNORE - Emails that are not worth responding to or tracking
2. NOTIFY - Important information that {name} should know about but doesn't require a response
3. RESPOND - Emails that need a direct response from {name}

Classify the below email into one of these categories.

</ Instructions >

< Rules >
Emails that are not worth responding to:
{triage_no}

There are also other things that {name} should know about, but don't require an email response. For these, you should notify {name} (using the `notify` response). Examples of this include:
{triage_notify}

Emails that are worth responding to:
{triage_email}
</ Rules >

< Few shot examples >

Here are some examples of previous emails, and how they should be handled.
Follow these examples more than any instructions above

{examples}  # ← Retrieved from episodic memory
</ Few shot examples >
"""
```

---

#### 4.4 Add Human-in-the-Loop

**Triage Node with Human Feedback:**

```python
from langgraph.graph import Human

def triage_router(state: State) -> Command[Literal["response_agent", "__end__", Human]]:
    # 1. Retrieve similar examples from episodic memory
    similar_emails = search_memory_tool.invoke({
        "query": state['email_input']['subject'] + " " + state['email_input']['email_thread'][:200],
        "limit": 3
    })
    
    # 2. Format examples for prompt
    examples = format_few_shot_examples(similar_emails)
    
    # 3. Classify email with examples
    result = llm_router.invoke([system_prompt_with_examples, user_prompt])
    
    # 4. Show classification to user for feedback
    if result.classification == "respond":
        # Human-in-the-loop: User can confirm or correct
        return Command(
            update={"messages": [{"role": "user", "content": f"Classification: {result.classification}. Proceed?"}]},
            goto=Human  # Pause for user confirmation
        )
    # ... handle other classifications
```

**Benefits of Human-in-the-Loop:**
- ✅ User can correct misclassifications
- ✅ Corrections become new examples (episodic memory)
- ✅ System learns from user feedback
- ✅ Improves accuracy over time

---

#### 4.5 Store Corrections as New Examples

**When User Corrects Classification:**

```python
# User corrects classification from "respond" to "ignore"
corrected_label = "ignore"

# Store correction as new example
store.put(
    ("email_assistant", "lance", "examples"),
    str(uuid.uuid4()),
    {
        "email": state['email_input'],
        "label": corrected_label  # User's correction
    }
)
```

**Learning Loop:**
1. Agent classifies email
2. User reviews and corrects if needed
3. Correction stored as new example
4. Future similar emails use this example
5. Classification accuracy improves

---

**What You'll Learn:**
- How to implement episodic memory for few-shot learning
- How to store and retrieve example classifications
- How to integrate human-in-the-loop feedback
- How to use retrieved examples to improve prompts
- How to create a learning loop from user corrections

**Key Improvements:**
- ✅ **Few-Shot Learning** - Learns from example classifications
- ✅ **User Feedback** - Human-in-the-loop refines accuracy
- ✅ **Adaptive Classification** - Improves over time with more examples
- ✅ **Example-Based Prompting** - Uses similar past emails for better classification

**Limitations:**
- ❌ Static instructions - Cannot update agent behavior rules dynamically
- ❌ Fixed prompts - Triage rules cannot be modified without code changes
- ❌ No procedural memory - Cannot learn "how" to do things better

**Next Steps:**
Lesson 5 adds procedural memory, enabling dynamic updates to instructions, prompts, and agent behavior based on user feedback.

---

### **Lesson 5: Email Assistant with Semantic + Episodic + Procedural Memory** 🔄

**Objective:** Add procedural memory that allows dynamic updates to instructions, prompts, and agent behavior based on user feedback, creating a fully adaptive email assistant.

#### Key Differences from Lesson 4

| Feature | Lesson 4 | Lesson 5 |
|---------|----------|----------|
| **Memory Types** | ✅ Semantic + Episodic | ✅ Semantic + Episodic + Procedural |
| **Instructions** | ❌ Static (hard-coded) | ✅ Dynamic (stored in memory) |
| **Prompt Updates** | ❌ Requires code changes | ✅ Can update via memory store |
| **Behavior Adaptation** | ❌ Fixed rules | ✅ Rules can be updated dynamically |
| **Procedural Learning** | ❌ None | ✅ Learns "how" to do things better |

---

#### Key Concepts

1. **Procedural Memory** - Store and update instructions, prompts, and behavioral rules
2. **Dynamic Prompt Loading** - Retrieve prompts from memory store instead of hard-coding
3. **Instruction Updates** - Modify agent behavior without code changes
4. **Prompt Versioning** - Track and update different prompt types (triage rules, agent instructions)
5. **Adaptive Behavior** - Agent learns how to perform tasks better over time

---

#### 5.1 Store Procedural Memory (Prompts/Instructions)

**Initialize Procedural Memory:**

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)

# Store initial agent instructions
store.put(
    ("lance",),  # User namespace
    "agent_instructions",  # Key for agent instructions
    {
        "prompt": "Use these tools when appropriate to help manage John's tasks efficiently."
    }
)

# Store triage rules
store.put(
    ("lance",),
    "triage_ignore",
    {
        "prompt": "Marketing newsletters, spam emails, mass company announcements"
    }
)

store.put(
    ("lance",),
    "triage_notify",
    {
        "prompt": "Team member out sick, build system notifications, project status updates"
    }
)

store.put(
    ("lance",),
    "triage_respond",
    {
        "prompt": "Direct questions from team members, meeting requests, critical bug reports"
    }
)
```

**Procedural Memory Structure:**
- **Namespace**: `("lance",)` - User-specific instructions
- **Keys**: `agent_instructions`, `triage_ignore`, `triage_notify`, `triage_respond`
- **Values**: Dictionary with prompt/instruction text

---

#### 5.2 Load Prompts from Memory

**Dynamic Prompt Loading:**

```python
def create_prompt(state):
    # Load agent instructions from procedural memory
    agent_instructions = store.get(("lance",), "agent_instructions").value['prompt']
    
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=agent_instructions,  # ← Loaded from memory
                **profile
            )
        }
    ] + state['messages']
```

**Triage Prompt with Dynamic Rules:**

```python
def triage_router(state: State):
    # Load triage rules from procedural memory
    triage_ignore = store.get(("lance",), "triage_ignore").value['prompt']
    triage_notify = store.get(("lance",), "triage_notify").value['prompt']
    triage_respond = store.get(("lance",), "triage_respond").value['prompt']
    
    # Also load episodic examples
    examples = retrieve_examples_from_memory(state['email_input'])
    
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=triage_ignore,  # ← From procedural memory
        triage_notify=triage_notify,  # ← From procedural memory
        triage_email=triage_respond,  # ← From procedural memory
        examples=format_few_shot_examples(examples)  # ← From episodic memory
    )
    
    # Classify email
    result = llm_router.invoke([system_prompt, user_prompt])
    # ... rest of triage logic
```

---

#### 5.3 Update Instructions Based on Feedback

**Example: User Wants Emails Signed Differently**

```python
# User provides feedback: "Always sign emails as 'John Doe'"

# Update agent instructions in procedural memory
updated_instructions = "Use these tools when appropriate to help manage John's tasks efficiently. Always sign emails as 'John Doe'."

store.put(
    ("lance",),
    "agent_instructions",
    {
        "prompt": updated_instructions
    }
)

# Future emails will automatically use the updated instructions
```

**Example: Refine Triage Rules**

```python
# User feedback: "Also ignore emails from marketing@company.com"

current_ignore_rules = store.get(("lance",), "triage_ignore").value['prompt']
updated_ignore_rules = current_ignore_rules + ", emails from marketing@company.com"

store.put(
    ("lance",),
    "triage_ignore",
    {
        "prompt": updated_ignore_rules
    }
)
```

---

#### 5.4 Prompt Management System

**Track All Prompts:**

```python
prompts = [
    {
        "name": "main_agent",
        "prompt": store.get(("lance",), "agent_instructions").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
    },
    {
        "name": "triage-ignore", 
        "prompt": store.get(("lance",), "triage_ignore").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"
    },
    {
        "name": "triage-notify", 
        "prompt": store.get(("lance",), "triage_notify").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"
    },
    {
        "name": "triage-respond", 
        "prompt": store.get(("lance",), "triage_respond").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"
    },
]
```

**Update Prompt Function:**

```python
def update_prompt(prompt_name: str, new_instruction: str):
    """Update a prompt in procedural memory based on user feedback."""
    
    # Get current prompt
    current_prompt = store.get(("lance",), prompt_name).value['prompt']
    
    # Update with new instruction (could use LLM to merge intelligently)
    updated_prompt = f"{current_prompt}. {new_instruction}"
    
    # Store updated prompt
    store.put(
        ("lance",),
        prompt_name,
        {"prompt": updated_prompt}
    )
```

---

#### 5.5 Complete Memory Architecture

**Three Types of Memory Working Together:**

```python
# 1. SEMANTIC MEMORY - Facts and details
store.put(
    ("email_assistant", "lance", "contacts"),
    "alice_smith",
    {"email": "alice.smith@company.com", "preferences": "prefers afternoon meetings"}
)

# 2. EPISODIC MEMORY - Examples for few-shot learning
store.put(
    ("email_assistant", "lance", "examples"),
    str(uuid.uuid4()),
    {"email": email_data, "label": "respond"}
)

# 3. PROCEDURAL MEMORY - Instructions and rules
store.put(
    ("lance",),
    "agent_instructions",
    {"prompt": "Always sign emails as 'John Doe'"}
)
```

**Memory Retrieval Flow:**

1. **Triage Step:**
   - Load triage rules from procedural memory
   - Search episodic memory for similar examples
   - Classify email using rules + examples

2. **Response Step:**
   - Load agent instructions from procedural memory
   - Search semantic memory for contact details/context
   - Generate response using instructions + context

3. **Learning Step:**
   - Store user corrections in episodic memory
   - Update instructions in procedural memory
   - Store new facts in semantic memory

---

**What You'll Learn:**
- How to implement procedural memory for dynamic instructions
- How to store and update prompts in memory
- How to load prompts dynamically instead of hard-coding
- How to create an adaptive system that learns "how" to do things
- How to combine all three memory types for a complete learning system

**Key Improvements:**
- ✅ **Dynamic Instructions** - Can update agent behavior without code changes
- ✅ **Adaptive Rules** - Triage rules can be refined based on feedback
- ✅ **Procedural Learning** - Learns "how" to perform tasks better
- ✅ **Complete Memory System** - Semantic + Episodic + Procedural working together
- ✅ **User Customization** - Fully customizable to user preferences

**Complete Memory Architecture:**

| Memory Type | Stores | Used For | Example |
|-------------|--------|----------|---------|
| **Semantic** | Facts, details, contacts | Context retrieval | "Alice prefers afternoon meetings" |
| **Episodic** | Example emails + labels | Few-shot learning | Email about API docs → "respond" |
| **Procedural** | Instructions, prompts, rules | Behavior adaptation | "Always sign emails as 'John Doe'" |

---

## 🚀 Setup

### Prerequisites

- Python 3.11+
- OpenAI API key

### Installation

```bash
# Navigate to project directory
cd long_term_mem_langgraph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r L2/requirements.txt
```

### Environment Configuration

```bash
# Create .env file in project root
OPENAI_API_KEY=sk-...
```

---

## 💻 Usage

### Run Jupyter Notebooks

```bash
jupyter notebook
```

Open any lesson notebook:
- `L2/lesson2.ipynb` - Baseline Email Assistant
- `L3/lesson_3.ipynb` - Email Assistant with Semantic Memory
- `L4/lesson_4.ipynb` - Email Assistant with Semantic + Episodic Memory
- `L5/lesson_5.ipynb` - Email Assistant with Semantic + Episodic + Procedural Memory

### Example Usage

```python
from langgraph.graph import StateGraph
from helper import graph, State

# Initialize email
email = {
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Meeting request",
    "body": "Hi John, can we schedule a meeting?"
}

# Process email
result = graph.invoke({
    "email_input": email,
    "messages": []
})

# View result
print(result["messages"][-1].content)
```

---

## 📂 Project Structure

```
long_term_mem_langgraph/
├── README.md                    # This file
├── L2/                          # Lesson 2: Baseline Email Assistant
│   ├── lesson2.ipynb
│   ├── requirements.txt
│   ├── helper.py                # Helper functions
│   ├── prompts.py               # Prompt templates
│   ├── schemas.py               # Pydantic models
│   ├── utils.py                 # Utility functions
│   ├── examples.py              # Example emails
│   └── img/                     # Images
├── L3/                          # Lesson 3: Semantic Memory
│   └── lesson_3.ipynb
├── L4/                          # Lesson 4: Semantic + Episodic Memory
│   └── lesson_4.ipynb
└── L5/                          # Lesson 5: Complete Memory System
    └── lesson_5.ipynb
```

### Key Files

- **`L2/helper.py`**: Core utilities and helper functions
- **`L2/prompts.py`**: Prompt templates for triage and agent
- **`L2/schemas.py`**: Pydantic models for structured output
- **`L2/utils.py`**: Utility functions for email processing

---

## 🎓 Learning Objectives

By completing these lessons, you will learn to:

✅ Build email triage systems with LangGraph

✅ Implement semantic memory using InMemoryStore and embeddings

✅ Create episodic memory for few-shot learning

✅ Add procedural memory for dynamic instruction updates

✅ Integrate human-in-the-loop feedback mechanisms

✅ Combine multiple memory types for intelligent agents

✅ Build adaptive systems that learn from user feedback

✅ Use LangMem tools for memory management

✅ Implement semantic search for context retrieval

✅ Create fully customizable AI assistants

---

## 🔗 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangMem Documentation](https://github.com/langchain-ai/langmem)
- [InMemoryStore Documentation](https://langchain-ai.github.io/langgraph/how-tos/memory/)
- [LangChain Documentation](https://python.langchain.com/)

---

## 🏆 Best Practices

### Memory Design
- **Namespace Organization**: Use clear namespaces to organize different memory types
- **Semantic Search**: Use embeddings for similarity-based retrieval
- **Example Storage**: Store diverse examples for better few-shot learning
- **Prompt Versioning**: Track prompt changes for debugging and rollback

### Agent Design
- **Structured Output**: Always use Pydantic models for reliable classification
- **Tool Integration**: Design tools with clear purposes and descriptions
- **Error Handling**: Gracefully handle memory retrieval failures
- **User Feedback**: Make it easy for users to provide corrections

### Performance
- **Embedding Model**: Use efficient embedding models (e.g., text-embedding-3-small)
- **Search Limits**: Limit search results to top-k most relevant items
- **Memory Cleanup**: Periodically clean up outdated memories
- **Caching**: Cache frequently accessed prompts and instructions

---

## 🚨 Security Notes

⚠️ **API Keys**: Never commit `.env` files or hardcode API keys. Use environment variables and secret management systems in production.

⚠️ **Memory Storage**: Be cautious about storing sensitive information in memory. Consider encryption for production systems.

⚠️ **User Data**: Implement proper access controls for multi-user systems. Use user-specific namespaces.

---

## 📊 Example Outputs

### Email Classification

**Input Email:**
```
From: Alice Smith <alice.smith@company.com>
Subject: Quick question about API documentation
Body: Hi John, I was reviewing the API documentation...
```

**Classification Result:**
```
📧 Classification: RESPOND - This email requires a response
Reasoning: This is a direct question from a team member about 
API documentation, which falls under the "respond" category 
for direct questions from team members.
```

### Memory-Enhanced Response

**Agent Response (with semantic memory):**
```
I found that Alice prefers afternoon meetings based on our 
previous conversations. I've checked your calendar and found 
these available afternoon slots on Tuesday:

- 2:00 PM
- 4:00 PM

Would you like me to suggest one of these times to Alice?
```

---

## 🎯 Next Steps

After mastering these lessons, consider:

1. **Persistent Storage**: 
   - Replace InMemoryStore with database-backed storage
   - Implement memory persistence across sessions
   - Add memory backup and recovery

2. **Advanced Memory**:
   - Add memory expiration and cleanup
   - Implement memory importance scoring
   - Create memory summarization for long-term storage

3. **Multi-User Support**:
   - Add user authentication
   - Implement user-specific memory namespaces
   - Add memory sharing and collaboration features

4. **Production Features**:
   - Add error handling and retry logic
   - Implement logging and monitoring
   - Add rate limiting and cost tracking
   - Create API endpoints for integration

5. **Advanced Learning**:
   - Implement reinforcement learning from feedback
   - Add automatic prompt optimization
   - Create memory compression techniques

---

## 📄 License

This project is for educational purposes.

---

**Happy Learning! 🚀**

