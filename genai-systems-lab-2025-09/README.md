# GenAI Systems Lab 2025-09 🚀

A comprehensive hands-on course covering the fundamentals and advanced topics in Generative AI systems, from basic chatbots to production-ready RAG applications, agentic AI, and evaluation frameworks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Course Structure](#course-structure)
- [Sessions](#sessions)
  - [Session 01: Your First Chatbot](#session-01-your-first-chatbot)
  - [Session 02: Python Fundamentals for GenAI](#session-02-python-fundamentals-for-genai)
  - [Session 03: Object-Oriented Programming & Local LLMs](#session-03-object-oriented-programming--local-llms)
  - [Session 04: Introduction to RAG](#session-04-introduction-to-rag)
  - [Session 06: Evaluation with RAGAS](#session-06-evaluation-with-ragas)
  - [Session 07: Fine-Tuning](#session-07-fine-tuning)
  - [Session 08: LangChain Framework](#session-08-langchain-framework)
  - [Session 09: Agentic AI with LangGraph](#session-09-agentic-ai-with-langgraph)
  - [Session 10: LangSmith Evaluation](#session-10-langsmith-evaluation)
  - [Session 11: Model Context Protocol (MCP)](#session-11-model-context-protocol-mcp)
  - [Session 07 Complete: Full-Stack Chatbot](#session-07-complete-full-stack-chatbot)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Key Technologies](#key-technologies)

---

## 🎯 Overview

This course provides a **hands-on, project-based learning experience** for building production-ready Generative AI systems. You'll progress from simple chatbots to sophisticated multi-agent systems with evaluation frameworks.

**Learning Path:**
```
Basic Chatbot → Python Basics → OOP & Local LLMs → RAG → Evaluation → Fine-Tuning → LangChain → Agentic AI → MCP → Full-Stack App
```

**Key Outcomes:**
- ✅ Build chatbots with OpenAI API
- ✅ Understand vector embeddings and semantic search
- ✅ Implement RAG (Retrieval-Augmented Generation) systems
- ✅ Evaluate LLM applications with RAGAS and LangSmith
- ✅ Fine-tune models for domain-specific tasks
- ✅ Build agentic AI systems with LangGraph
- ✅ Create production-ready full-stack applications

---

## 🛠️ Prerequisites

- **Python 3.10+** installed
- **Basic Python knowledge** (variables, functions, loops)
- **OpenAI API Key** (for cloud-based LLMs)
- **Ollama** installed (for local LLMs)
- **Milvus** running locally (for vector database)
- **Node.js** installed (for full-stack chatbot)

---

## 📚 Course Structure

| Session | Topic | Focus | Duration |
|---------|-------|-------|----------|
| **01** | First Chatbot | OpenAI API basics, moderation | 1-2 hours |
| **02** | Python Fundamentals | Lists, dicts, exception handling | 2-3 hours |
| **03** | OOP & Local LLMs | Classes, inheritance, Ollama | 2-3 hours |
| **04** | RAG Introduction | Embeddings, vector search, Milvus | 3-4 hours |
| **06** | Evaluation | RAGAS metrics, retrieval evaluation | 2-3 hours |
| **07** | Fine-Tuning | OpenAI fine-tuning API | 2-3 hours |
| **08** | LangChain | Chunking, chains, RAG pipeline | 3-4 hours |
| **09** | Agentic AI | LangGraph, multi-agent systems | 4-5 hours |
| **10** | LangSmith | Tracing, evaluation, datasets | 2-3 hours |
| **11** | MCP | Model Context Protocol | 2-3 hours |
| **07 Complete** | Full-Stack App | React frontend + Python backend | 4-5 hours |

---

## 📝 Sessions

### **Session 01: Your First Chatbot** 🤖

**Objective:** Build your first interactive chatbot using OpenAI's API.

#### Key Concepts
- OpenAI API integration
- Chat completions (`chat.completions.create`)
- System and user messages
- Response moderation
- Environment variable management

#### Files
- `first_chatbot.ipynb` - Interactive chatbot loop
- `hello_world.ipynb` - Basic API connection
- `input_prompt.ipynb` - User input handling
- `response_moderation.ipynb` - Content moderation

#### Example Code

**Basic Chatbot:**
```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_question(prompt):
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Interactive loop
import time
while True:
    user_prompt = input("Ask something: ")
    if user_prompt.lower() != 'quit':
        response = ask_question(user_prompt)
        print("\nOpenAI says:", response)
        time.sleep(3)
    else:
        break
```

**Key Takeaways:**
- ✅ Understand OpenAI API structure
- ✅ Handle user input and responses
- ✅ Implement conversation loops
- ✅ Add content moderation

---

### **Session 02: Python Fundamentals for GenAI** 🐍

**Objective:** Master Python data structures and patterns commonly used in GenAI applications.

#### Key Concepts
- Lists and list comprehensions
- Dictionaries and dictionary operations
- Exception handling (try/except)
- Dictionary usage in GenAI (messages, responses)
- OpenAI integration patterns

#### Files
- `1_list_examples.ipynb` - List operations
- `2_dictionary_examples.ipynb` - Dictionary patterns
- `3_exception_handling.ipynb` - Error handling
- `4_dictionary_in_genai.ipynb` - GenAI-specific dict usage
- `5_openai_integration.ipynb` - OpenAI API patterns
- `6_openai_modeartion_integration.ipynb` - Moderation API
- `7_hello.py` - Python script example

#### Example Code

**Dictionary in GenAI Context:**
```python
# Message structure for OpenAI API
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]

# Response structure
response = {
    "id": "chatcmpl-123",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "Python is a programming language..."
        }
    }],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 20,
        "total_tokens": 35
    }
}
```

**Exception Handling:**
```python
def safe_api_call(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I encountered an error."
```

**Key Takeaways:**
- ✅ Master Python data structures
- ✅ Handle errors gracefully
- ✅ Understand GenAI API data formats
- ✅ Build robust error handling

---

### **Session 03: Object-Oriented Programming & Local LLMs** 🏗️

**Objective:** Learn OOP principles and run LLMs locally with Ollama.

#### Key Concepts
- Classes and objects
- Inheritance
- Encapsulation
- Polymorphism
- Local LLM deployment (Ollama)
- Comparing local vs. cloud LLMs

#### Files
- `1_class_objects.py` - Basic classes
- `2_inheritance.py` - Class inheritance
- `3_encapsulation.py` - Data encapsulation
- `4_polymorphism.py` - Polymorphic behavior
- `5_local_llm.ipynb` - Ollama integration
- `6_local_llm_vs_openai.ipynb` - Performance comparison

#### Example Code

**Basic Class:**
```python
class ChatBot:
    def __init__(self, model="gpt-5-nano"):
        self.model = model
        self.conversation_history = []
    
    def ask(self, prompt):
        response = client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history + [
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer
```

**Local LLM with Ollama:**
```python
import ollama
import time

def ask_question_local_llm(prompt):
    response = ollama.chat(
        model='llama3',
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content']

# Usage
start = time.time()
response = ask_question_local_llm("What is Python?")
end = time.time()
print(f"Response: {response}")
print(f"Time taken: {end - start} seconds")
```

**Key Takeaways:**
- ✅ Understand OOP principles
- ✅ Design reusable class structures
- ✅ Run LLMs locally (privacy, cost benefits)
- ✅ Compare local vs. cloud LLM performance

---

### **Session 04: Introduction to RAG** 🔍

**Objective:** Build your first RAG (Retrieval-Augmented Generation) system using vector embeddings and semantic search.

#### Key Concepts
- Vector embeddings (sentence transformers)
- Semantic similarity search
- Document chunking and indexing
- Milvus vector database
- RAG pipeline architecture

#### Files
- `rag_intro.ipynb` - Basic RAG with sentence transformers
- `rag_intro_employee.ipynb` - Employee policy RAG system
- `milvus_rag.ipynb` - Milvus integration
- `milvus_rag_search.ipynb` - Advanced search patterns
- `llm_utlity.py` - Utility functions

#### Example Code

**Basic RAG with Sentence Transformers:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Sample documents
documents = [
    {"section": "Pay Policies", "content": "Employees are paid bi-weekly via direct deposit."},
    {"section": "Internet Use", "content": "Company internet must be used for work-related tasks only."},
    {"section": "Break at Work", "content": "Employees can take an hour break."},
]

# Step 1: Encode documents
model = SentenceTransformer("all-MiniLM-L6-v2")
content_corpus = [doc["content"] for doc in documents]
doc_vectors = model.encode(content_corpus)

# Step 2: Encode query
query = "What's the internet usage policy?"
query_vec = model.encode([query])[0]

# Step 3: Find most similar documents
similarities = model.similarity(query_vec, doc_vectors)
similarities = np.asarray(similarities).squeeze()

# Step 4: Get top 3 most relevant
top_3_indices = np.argsort(similarities)[::-1][:3]
top_docs = [documents[i]['content'] for i in top_3_indices]

# Step 5: Use retrieved context with LLM
context = "\n---\n".join(top_docs)
prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
answer = llm.generate(prompt)
```

**Milvus RAG:**
```python
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Create collection
collection = Collection("employee_policies")

# Insert documents
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode([doc["content"] for doc in documents])
collection.insert([vectors, documents])

# Search
query_vec = model.encode([query])[0]
results = collection.search(
    data=[query_vec],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=3
)
```

**Key Takeaways:**
- ✅ Understand vector embeddings and semantic search
- ✅ Build end-to-end RAG pipelines
- ✅ Use Milvus for production vector storage
- ✅ Implement context retrieval for LLMs

---

### **Session 06: Evaluation with RAGAS** 📊

**Objective:** Systematically evaluate RAG systems using RAGAS (Retrieval-Augmented Generation Assessment) framework.

#### Key Concepts
- RAGAS evaluation metrics
- Faithfulness (answer grounded in context)
- Answer correctness
- Context relevance
- Retrieval evaluation
- LLM generation evaluation

#### Files
- `eval_llmgeneration_with_ragas.ipynb` - LLM generation evaluation
- `eval_retrieval_with_ragas.ipynb` - Retrieval evaluation
- `milvus_chatbot_with_rag.py` - RAG chatbot implementation
- `milvus_rag_data_setup.py` - Data preparation script

#### Example Code

**LLM Generation Evaluation:**
```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Reference data
reference_data = [
    {
        "question": "What is the company's policy on remote work?",
        "ground_truth": "Remote work is allowed up to 3 days per week.",
        "context": "Remote work is allowed up to 3 days per week."
    }
]

# Perform retrieval and generation
def perform_retrieval(question):
    retrieved_context = search_similar(question, "employee_policies", 1)[0]['content']
    return retrieved_context

def generate_answer(question, context):
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "Answer based on context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

# Build evaluation dataset
question = reference_data[0]['question']
retrieved_context = [perform_retrieval(question)]
llm_answer = generate_answer(question, retrieved_context[0])

dataset_dict = {
    "question": [question],
    "contexts": [retrieved_context],  # List of strings
    "ground_truth": [reference_data[0]['ground_truth']],
    "answer": [llm_answer]
}

dataset = Dataset.from_dict(dataset_dict)

# Evaluate
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_correctness]
)

print(f"Faithfulness: {result['faithfulness']}")
print(f"Answer Correctness: {result['answer_correctness']}")
```

**Retrieval Evaluation:**
```python
from ragas.metrics import context_precision, context_recall

# Evaluate retrieval quality
retrieval_result = evaluate(
    dataset=dataset,
    metrics=[context_precision, context_recall]
)

print(f"Context Precision: {retrieval_result['context_precision']}")
print(f"Context Recall: {retrieval_result['context_recall']}")
```

**Key Takeaways:**
- ✅ Understand RAG evaluation metrics
- ✅ Measure retrieval quality
- ✅ Measure generation quality
- ✅ Build evaluation pipelines

---

### **Session 07: Fine-Tuning** 🎯

**Objective:** Fine-tune OpenAI models for domain-specific tasks.

#### Key Concepts
- Fine-tuning vs. prompt engineering
- Training data preparation (JSONL format)
- Fine-tuning job creation
- Model evaluation
- Fine-tuned model deployment

#### Files
- `fine_tuning.ipynb` - Fine-tuning workflow
- `data/training_data.jsonl` - Training dataset
- `data/validation_data.jsonl` - Validation dataset

#### Example Code

**Prepare Training Data (JSONL format):**
```jsonl
{"messages": [{"role": "system", "content": "You are a brand voice assistant."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hey there! How can I help?"}]}
{"messages": [{"role": "system", "content": "You are a brand voice assistant."}, {"role": "user", "content": "What's your return policy?"}, {"role": "assistant", "content": "We offer 30-day returns, no questions asked!"}]}
```

**Fine-Tuning Workflow:**
```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Upload training files
with open("data/training_data.jsonl", "rb") as f:
    training_file = client.files.create(file=f, purpose="fine-tune")

with open("data/validation_data.jsonl", "rb") as f:
    validation_file = client.files.create(file=f, purpose="fine-tune")

print(f"Training file ID: {training_file.id}")
print(f"Validation file ID: {validation_file.id}")

# Step 2: Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,
    model="gpt-4.1-2025-04-14",
    suffix="brand-voice-support"  # Optional model name suffix
)

print(f"Fine-tune job created: {job.id}")

# Step 3: Monitor job status
import time
while True:
    job_status = client.fine_tuning.jobs.retrieve(job.id)
    print(f"Status: {job_status.status}")
    
    if job_status.status == "succeeded":
        print(f"Fine-tuned model: {job_status.fine_tuned_model}")
        break
    elif job_status.status == "failed":
        print("Fine-tuning failed!")
        break
    
    time.sleep(60)  # Check every minute

# Step 4: Use fine-tuned model
def ask_with_finetuned_model(prompt):
    response = client.chat.completions.create(
        model=job_status.fine_tuned_model,  # Use fine-tuned model
        messages=[
            {"role": "system", "content": "You are a brand voice assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
```

**Key Takeaways:**
- ✅ Understand when to fine-tune vs. prompt
- ✅ Prepare training data in correct format
- ✅ Create and monitor fine-tuning jobs
- ✅ Deploy fine-tuned models

---

### **Session 08: LangChain Framework** 🔗

**Objective:** Build production-ready RAG systems using LangChain framework.

#### Key Concepts
- LangChain document loaders
- Text splitting strategies
- Vector stores (Milvus integration)
- Retrieval chains
- Prompt templates
- LangChain RAG pipeline

#### Files
- `langchain_chunking.ipynb` - Document chunking strategies
- `langchain_llmchain.ipynb` - LLM chains
- `langchain_rag.ipynb` - Complete RAG pipeline
- `data/Motor_Vehicle_Claim_Complex.pdf` - Sample document

#### Example Code

**LangChain RAG Pipeline:**
```python
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Step 1: Create sample documents
texts = [
    "Milvus is a vector database designed for scalable similarity search.",
    "Retrieval-Augmented Generation combines retrieval with generation.",
    "LangChain provides tools for building RAG pipelines easily.",
]

# Step 2: Create embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Milvus.from_texts(
    texts,
    embedding=embeddings,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name="rag_docs"
)

# Step 3: Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Step 4: Test retrieval
query = "What is Milvus used for?"
results = retriever.get_relevant_documents(query)

for i, doc in enumerate(results):
    print(f"\n--- Retrieved Document {i+1} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)

# Step 5: Create LLM
llm = ChatOpenAI(model="gpt-5-mini")

# Step 6: Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Step 7: Query
result = qa_chain.invoke({"query": "What is Milvus?"})
print(f"Answer: {result['result']}")
print(f"Sources: {result['source_documents']}")
```

**Document Chunking:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = splitter.split_text(long_document)
print(f"Created {len(chunks)} chunks")
```

**Key Takeaways:**
- ✅ Understand LangChain architecture
- ✅ Build RAG pipelines with LangChain
- ✅ Implement document chunking strategies
- ✅ Integrate with vector databases

---

### **Session 09: Agentic AI with LangGraph** 🤖

**Objective:** Build multi-agent systems using LangGraph for complex workflows.

#### Key Concepts
- LangGraph state management
- Multi-agent workflows
- Sequential and parallel nodes
- Conditional routing
- State sharing between agents
- Agent orchestration

#### Files
- `library_agentic_ai_application.ipynb` - Basic agentic system
- `library_agentic_ai_application_with_llm.ipynb` - LLM-powered agents
- `library_agentic_ai_application_llm_langsmith.ipynb` - With LangSmith tracing
- `graph_sequential_nodes.ipynb` - Sequential workflow
- `graph_parallel_nodes.ipynb.ipynb` - Parallel execution
- `graph_with_state.ipynb` - State management
- `library_agentic_ai.py` - Library agent implementation
- `agents/AgentA.py` - Custom agent example

#### Example Code

**Basic LangGraph Workflow:**
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Define shared state
class LibraryState(TypedDict):
    question: str
    faq_answer: str
    checkout_info: str
    final_answer: str

# Define agents (nodes)
def ClassifierAgent(state: LibraryState):
    q = state["question"].lower()
    
    if "available" in q or "checkout" in q:
        return {"faq_answer": "", "checkout_info": "Book available: The Hobbit"}
    else:
        return {"faq_answer": "Library opens at 9 AM", "checkout_info": ""}

def FAQAgent(state: LibraryState):
    if not state.get("faq_answer"):
        return {"faq_answer": "Default FAQ: Library rules apply"}
    return {}

def CheckoutAgent(state: LibraryState):
    if not state.get("checkout_info"):
        return {"checkout_info": "Checkout info: Not requested"}
    return {}

def ResponseAgent(state: LibraryState):
    final = f"Q: {state['question']}\n"
    if state.get("faq_answer"):
        final += f"FAQ: {state['faq_answer']}\n"
    if state.get("checkout_info"):
        final += f"Checkout: {state['checkout_info']}"
    return {"final_answer": final}

# Build the graph
builder = StateGraph(LibraryState)
builder.add_node("ClassifierAgent", ClassifierAgent)
builder.add_node("FAQAgent", FAQAgent)
builder.add_node("CheckoutAgent", CheckoutAgent)
builder.add_node("ResponseAgent", ResponseAgent)

# Define edges
builder.add_edge(START, "ClassifierAgent")
builder.add_edge("ClassifierAgent", "FAQAgent")
builder.add_edge("ClassifierAgent", "CheckoutAgent")
builder.add_edge("FAQAgent", "ResponseAgent")
builder.add_edge("CheckoutAgent", "ResponseAgent")
builder.add_edge("ResponseAgent", END)

# Compile and run
graph = builder.compile()
print(graph.get_graph().draw_ascii())

result = graph.invoke({"question": "Is The Hobbit available?"})
print("\n--- Final Answer ---")
print(result["final_answer"])
```

**LLM-Powered Agents:**
```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

llm = ChatOpenAI(model="gpt-5-nano")

def LLMClassifierAgent(state: LibraryState):
    response = llm.invoke([
        {"role": "system", "content": "Classify the question."},
        {"role": "user", "content": state["question"]}
    ])
    # Parse response and update state
    return {"classification": response.content}

# Add LLM agent to graph
builder.add_node("LLMClassifierAgent", LLMClassifierAgent)
```

**Key Takeaways:**
- ✅ Understand agentic AI architecture
- ✅ Build multi-agent workflows
- ✅ Implement state management
- ✅ Create sequential and parallel workflows

---

### **Session 10: LangSmith Evaluation** 📈

**Objective:** Monitor, trace, and evaluate LLM applications using LangSmith platform.

#### Key Concepts
- LangSmith tracing
- Function decorators (`@traceable`)
- Dataset creation and management
- Evaluation runs
- Performance monitoring
- Debugging LLM applications

#### Files
- `langsmith_setup.ipynb` - LangSmith configuration
- `rag_intro_employee.ipynb` - RAG with LangSmith tracing

#### Example Code

**LangSmith Setup:**
```python
# Add to .env file
# LANGCHAIN_PROJECT="Langsmith_Eval"
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
# LANGCHAIN_API_KEY=<your_langsmith_api_key>
```

**Enable Tracing:**
```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable

load_dotenv(override=True)
llm = ChatOpenAI(model="gpt-5-nano", api_key=os.getenv("OPENAI_API_KEY"))

# Enable tracing with one decorator
@traceable
def ask_question(user_prompt) -> str:
    """LLM function under test."""
    if isinstance(user_prompt, dict):
        user_prompt = user_prompt.get("input", "")

    system_msg = "You are a helpful assistant. Provide answer in one line."
    response = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ])
    
    return response.content

# Function calls are automatically traced
ask_question("Where is the museum Louvre?")
# View traces at https://smith.langchain.com/
```

**Create Dataset:**
```python
from langsmith import Client

client = Client()

# Create dataset
dataset = client.create_dataset(dataset_name="LangSmith-QA-Dataset")

# Add examples
client.create_examples(
    inputs=[
        {"input": "What is Python?"},
        {"input": "Explain RAG."}
    ],
    outputs=[
        {"output": "Python is a programming language."},
        {"output": "RAG is Retrieval-Augmented Generation."}
    ],
    dataset_id=dataset.id
)
```

**Key Takeaways:**
- ✅ Understand LangSmith platform
- ✅ Enable tracing for LLM applications
- ✅ Create and manage evaluation datasets
- ✅ Monitor application performance

---

### **Session 11: Model Context Protocol (MCP)** 🔌

**Objective:** Build MCP servers and clients for tool integration with LLMs.

#### Key Concepts
- Model Context Protocol (MCP) architecture
- FastMCP framework
- MCP server implementation
- MCP client integration
- Tool discovery and invocation
- Agent-tool communication

#### Files
- `FastMCP_Server_Client_Demo.ipynb` - MCP demo
- `mcp_server_fastmcp.py` - MCP server implementation
- `data.csv` - Sample dataset

#### Example Code

**MCP Server:**
```python
# mcp_server_fastmcp.py
from fastmcp import FastMCP
import pandas as pd

mcp = FastMCP("CSV Tools Server")

# Load data
df = pd.read_csv("data.csv")

@mcp.tool()
def summarize() -> str:
    """Get summary statistics of the dataset."""
    return f"Dataset has {len(df)} rows and {len(df.columns)} columns."

@mcp.tool()
def query(expr: str) -> str:
    """Query the dataset using pandas expression."""
    try:
        result = df.query(expr)
        return result.to_string()
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    mcp.run()
```

**MCP Client:**
```python
from fastmcp import Client
from fastmcp.client import PythonStdioTransport
import asyncio

async def main():
    transport = PythonStdioTransport("mcp_server_fastmcp.py")
    async with Client(transport) as client:
        # Discover tools
        tools = await client.list_tools()
        print("Tools:", [t.name for t in tools])
        
        # Call tool
        result = await client.call_tool("summarize", {})
        print("Result:", result)

await main()
```

**Agent Integration:**
```python
def decide_tool(text: str):
    text = text.lower()
    if "summarize" in text or "overview" in text:
        return "summarize", {}
    if "west" in text:
        return "query", {"expr": "region == 'West' and sales > 1500"}
    return "summarize", {}

async def run_agent(user_input, client):
    tool, params = decide_tool(user_input)
    print(f"Agent decided to use '{tool}'")
    
    result = await client.call_tool(tool, params)
    print("Result:", result)
```

**Key Takeaways:**
- ✅ Understand MCP architecture
- ✅ Build MCP servers with FastMCP
- ✅ Create MCP clients
- ✅ Integrate tools with LLM agents

---

### **Session 07 Complete: Full-Stack Chatbot** 🌐

**Objective:** Build a complete full-stack chatbot application with React frontend and Python backend.

#### Key Concepts
- RESTful API design
- FastAPI backend
- React frontend
- WebSocket communication
- State management
- Frontend-backend integration

#### Files
- `backend/main_backend.py` - FastAPI backend server
- `frontend/src/App.js` - React main component
- `frontend/src/components/ChatBox.js` - Chat interface component
- `frontend/package.json` - Node.js dependencies

#### Architecture

```
┌─────────────┐         HTTP/WebSocket          ┌─────────────┐
│   React     │  <──────────────────────────>  │   FastAPI   │
│  Frontend   │                                 │   Backend   │
└─────────────┘                                 └─────────────┘
                                                       │
                                                       ▼
                                                ┌─────────────┐
                                                │   OpenAI    │
                                                │     API     │
                                                └─────────────┘
```

#### Example Code

**Backend (FastAPI):**
```python
# backend/main_backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/chat")
async def chat(message: dict):
    user_message = message.get("message", "")
    
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
    )
    
    return {"response": response.choices[0].message.content}
```

**Frontend (React):**
```javascript
// frontend/src/components/ChatBox.js
import React, { useState } from 'react';
import './ChatBox.css';

function ChatBox() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMessage = { role: 'user', content: input };
        setMessages([...messages, userMessage]);
        setInput('');

        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: input })
        });

        const data = await response.json();
        setMessages([...messages, userMessage, 
                    { role: 'assistant', content: data.response }]);
    };

    return (
        <div className="chatbox">
            <div className="messages">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                        {msg.content}
                    </div>
                ))}
            </div>
            <div className="input-area">
                <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                />
                <button onClick={sendMessage}>Send</button>
            </div>
        </div>
    );
}

export default ChatBox;
```

**Key Takeaways:**
- ✅ Build RESTful APIs with FastAPI
- ✅ Create React frontends
- ✅ Integrate frontend and backend
- ✅ Deploy full-stack applications

---

## 🚀 Setup Instructions

### 1. Clone Repository
```bash
git clone <repository_url>
cd genai-systems-lab-2025-09
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install openai python-dotenv sentence-transformers
pip install pymilvus langchain langchain-openai langchain-community
pip install ragas datasets fastapi uvicorn
pip install ollama fastmcp
```

### 4. Setup Environment Variables
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPEN_AI_API_KEY=your_openai_api_key_here  # Some notebooks use this
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT="Langsmith_Eval"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```

### 5. Install Ollama (for Local LLMs)
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download
```

### 6. Install Milvus (for Vector Database)
```bash
# Using Docker
docker pull milvusdb/milvus:latest
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
```

### 7. Setup Frontend (for Session 07 Complete)
```bash
cd Session07_complete_chatbot/frontend
npm install
npm start
```

### 8. Setup Backend (for Session 07 Complete)
```bash
cd Session07_complete_chatbot/backend
pip install fastapi uvicorn python-dotenv openai
uvicorn main_backend:app --reload
```

---

## 📂 Project Structure

```
genai-systems-lab-2025-09/
├── README.md                          # This file
├── hello_world.ipynb                  # Quick start
├── .env                               # Environment variables (create this)
│
├── Session_01/                        # First Chatbot
│   ├── first_chatbot.ipynb
│   ├── hello_world.ipynb
│   ├── input_prompt.ipynb
│   └── response_moderation.ipynb
│
├── Session_02/                        # Python Fundamentals
│   ├── 1_list_examples.ipynb
│   ├── 2_dictionary_examples.ipynb
│   ├── 3_exception_handling.ipynb
│   ├── 4_dictionary_in_genai.ipynb
│   ├── 5_openai_integration.ipynb
│   ├── 6_openai_modeartion_integration.ipynb
│   └── 7_hello.py
│
├── Session_03/                        # OOP & Local LLMs
│   ├── 1_class_objects.py
│   ├── 2_inheritance.py
│   ├── 3_encapsulation.py
│   ├── 4_polymorphism.py
│   ├── 5_local_llm.ipynb
│   └── 6_local_llm_vs_openai.ipynb
│
├── Session_04/                        # RAG Introduction
│   ├── rag_intro.ipynb
│   ├── rag_intro_employee.ipynb
│   ├── milvus_rag.ipynb
│   ├── milvus_rag_search.ipynb
│   └── llm_utlity.py
│
├── Session_06/                        # Evaluation
│   ├── eval_llmgeneration_with_ragas.ipynb
│   ├── eval_retrieval_with_ragas.ipynb
│   ├── milvus_chatbot_with_rag.py
│   └── milvus_rag_data_setup.py
│
├── Session_07_Fine_Tuning/            # Fine-Tuning
│   ├── fine_tuning.ipynb
│   └── data/
│       ├── training_data.jsonl
│       └── validation_data.jsonl
│
├── Session_08_langchain/               # LangChain
│   ├── langchain_chunking.ipynb
│   ├── langchain_llmchain.ipynb
│   ├── langchain_rag.ipynb
│   └── data/
│       └── Motor_Vehicle_Claim_Complex.pdf
│
├── Session_09_AgenticAI/               # Agentic AI
│   ├── library_agentic_ai_application.ipynb
│   ├── library_agentic_ai_application_with_llm.ipynb
│   ├── library_agentic_ai_application_llm_langsmith.ipynb
│   ├── graph_sequential_nodes.ipynb
│   ├── graph_parallel_nodes.ipynb.ipynb
│   ├── graph_with_state.ipynb
│   ├── library_agentic_ai.py
│   └── agents/
│       └── AgentA.py
│
├── Session_10_Lansmith_Eval/           # LangSmith
│   ├── langsmith_setup.ipynb
│   └── rag_intro_employee.ipynb
│
├── Session_11_MCP/                     # Model Context Protocol
│   ├── FastMCP_Server_Client_Demo.ipynb
│   ├── mcp_server_fastmcp.py
│   └── data.csv
│
└── Session07_complete_chatbot/         # Full-Stack App
    ├── backend/
    │   └── main_backend.py
    └── frontend/
        ├── src/
        │   ├── App.js
        │   └── components/
        │       └── ChatBox.js
        └── package.json
```

---

## 🔧 Key Technologies

| Technology | Purpose | Used In |
|------------|---------|---------|
| **OpenAI API** | Cloud LLM service | Sessions 01, 02, 04, 06, 07, 08, 10 |
| **Ollama** | Local LLM deployment | Session 03 |
| **Sentence Transformers** | Vector embeddings | Session 04 |
| **Milvus** | Vector database | Sessions 04, 06 |
| **LangChain** | LLM framework | Sessions 08, 10 |
| **LangGraph** | Agent orchestration | Session 09 |
| **RAGAS** | RAG evaluation | Session 06 |
| **LangSmith** | LLM observability | Sessions 09, 10 |
| **FastMCP** | MCP framework | Session 11 |
| **FastAPI** | Python web framework | Session 07 Complete |
| **React** | Frontend framework | Session 07 Complete |

---

## 🎓 Learning Outcomes

By completing this course, you will be able to:

✅ **Build chatbots** using OpenAI API with moderation

✅ **Master Python** fundamentals essential for GenAI development

✅ **Understand OOP** principles and local LLM deployment

✅ **Implement RAG systems** with vector embeddings and semantic search

✅ **Evaluate LLM applications** using RAGAS and LangSmith

✅ **Fine-tune models** for domain-specific tasks

✅ **Build production RAG pipelines** with LangChain

✅ **Create agentic AI systems** with LangGraph

✅ **Monitor and debug** LLM applications with LangSmith

✅ **Integrate tools** with LLMs using MCP

✅ **Deploy full-stack** chatbot applications

---

## 📚 Additional Resources

- [OpenAI Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Milvus Documentation](https://milvus.io/docs)
- [RAGAS Documentation](https://docs.ragas.io/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Ollama Documentation](https://ollama.com/docs)

---

## 🚨 Common Issues & Solutions

### Issue: OpenAI API Key Not Found
**Solution:** Ensure `.env` file exists with `OPENAI_API_KEY` or `OPEN_AI_API_KEY` set correctly.

### Issue: Milvus Connection Failed
**Solution:** Ensure Milvus is running: `docker ps | grep milvus`

### Issue: Ollama Model Not Found
**Solution:** Pull the model first: `ollama pull llama3`

### Issue: Port Already in Use (FastAPI)
**Solution:** Change port: `uvicorn main_backend:app --port 8001`

### Issue: Frontend Dependencies Install Failed
**Solution:** Clear cache: `npm cache clean --force` then `npm install`

---

## 🎯 Next Steps

After completing this course:

1. **Build Your Own RAG System** - Apply concepts to your domain
2. **Deploy to Production** - Use cloud services (AWS, GCP, Azure)
3. **Experiment with Other Frameworks** - Try LlamaIndex, Haystack
4. **Explore Advanced Topics** - Multi-modal RAG, Fine-tuning optimization
5. **Contribute to Open Source** - LangChain, LangGraph, RAGAS

---

## 📄 License

This course material is for educational purposes.

---

**Happy Learning! 🚀**

