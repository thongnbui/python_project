# Functions, Tools, and Agents with LangChain

A comprehensive tutorial series on building AI agents using OpenAI function calling, LangChain tools, and conversational agents. Learn how to create structured outputs, extract information, and build intelligent routing systems.

**Course URL:** https://learn.deeplearning.ai/courses/functions-tools-agents-langchain

## Overview

This project demonstrates how to leverage OpenAI's function calling capabilities with LangChain to build sophisticated AI applications. You'll learn to create type-safe interactions with LLMs, extract structured data, integrate external APIs, and build conversational agents that can use multiple tools intelligently.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Lesson Overview](#lesson-overview)
- [Key Concepts](#key-concepts)
- [Getting Started](#getting-started)
- [Technologies Used](#technologies-used)
- [Use Cases](#use-cases)
- [Best Practices](#best-practices)

## Prerequisites

- Python 3.9+
- Basic understanding of LangChain
- Familiarity with OpenAI API
- Understanding of type hints and Python classes
- OpenAI API key

### Required Packages

```bash
pip install langchain openai pydantic python-dotenv requests wikipedia
```

## Project Structure

```
functions-tools-agents-langchain/
├── L3-function-calling-student.ipynb        # OpenAI function calling with Pydantic
├── L4-tagging-and-extraction-student.ipynb  # Tagging and extracting structured data
├── L5-tools-routing-apis-student.ipynb      # Building tools and routing to APIs
├── L6-functional_conversation-student.ipynb # Conversational agents with tools
└── README.md                                # This file
```

## Lesson Overview

### Lesson 3: OpenAI Function Calling

**Objective:** Master OpenAI function calling using Pydantic models for type-safe LLM interactions.

**Topics Covered:**
- Pydantic data classes for validation
- Converting Pydantic models to OpenAI function definitions
- Type safety and automatic validation
- Creating function schemas with descriptions

**Key Concepts:**

**Standard Python Class (No Validation):**
```python
class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

# No validation - accepts wrong types!
foo = User(name="Joe", age="bar", email="joe@gmail.com")  # age should be int
```

**Pydantic Class (With Validation):**
```python
from pydantic import BaseModel, Field

class pUser(BaseModel):
    name: str
    age: int
    email: str

# Validation error - won't accept string for age!
foo_p = pUser(name="Jane", age="bar", email="jane@gmail.com")  # Raises ValidationError
```

**Creating OpenAI Functions:**
```python
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")

weather_function = convert_pydantic_to_openai_function(WeatherSearch)
```

**Key Learning Points:**
- Pydantic provides automatic type validation
- Docstrings become function descriptions for LLMs
- Field descriptions help LLMs understand parameters
- Type-safe interactions prevent runtime errors
- Nested models enable complex data structures

### Lesson 4: Tagging and Extraction

**Objective:** Use OpenAI functions to tag content and extract structured information from unstructured text.

**Topics Covered:**
- Content tagging with sentiment and language detection
- Information extraction from natural language
- Building extraction chains with LangChain
- Parsing function outputs to JSON
- Handling optional fields and partial information

**Tagging Example:**
```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")

model = ChatOpenAI(temperature=0)
tagging_functions = [convert_pydantic_to_openai_function(Tagging)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])

model_with_functions = model.bind(
    functions=tagging_functions,
    function_call={"name": "Tagging"}
)

tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()

# Results: {"sentiment": "pos", "language": "en"}
tagging_chain.invoke({"input": "I love langchain"})

# Results: {"sentiment": "neg", "language": "it"}
tagging_chain.invoke({"input": "non mi piace questo cibo"})
```

**Extraction Example:**
```python
from typing import Optional, List

class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")

class Information(BaseModel):
    """Information to extract."""
    people: List[Person] = Field(description="List of info about people")

extraction_functions = [convert_pydantic_to_openai_function(Information)]
extraction_model = model.bind(
    functions=extraction_functions, 
    function_call={"name": "Information"}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])

extraction_chain = prompt | extraction_model

# Extracts: {"people": [{"name": "Joe", "age": 30}, {"name": "Martha", "age": null}]}
extraction_chain.invoke("Joe is 30, his mom is Martha")
```

**Use Cases:**
- Sentiment analysis and language detection
- Entity extraction from documents
- Contact information parsing
- Resume/CV data extraction
- Product review analysis
- Customer feedback categorization

### Lesson 5: Tools and Routing

**Objective:** Create custom tools and route LLM requests to appropriate APIs and functions.

**Topics Covered:**
- Creating tools with the `@tool` decorator
- Defining tool schemas with Pydantic
- Integrating external APIs (weather, Wikipedia)
- Tool naming and description best practices
- Formatting tools for OpenAI function calling

**Basic Tool Creation:**
```python
from langchain.agents import tool

@tool
def search(query: str) -> str:
    """Search for weather online"""
    return "42f"

# Automatic properties from decorator
search.name          # "search"
search.description   # "Search for weather online"
search.args          # {"query": {"type": "string"}}
```

**Advanced Tool with Schema:**
```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Thing to search for")

@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for the weather online."""
    return "42f"

# Enhanced schema with better descriptions
search.args  # More detailed parameter information
```

**Real-World API Integration (Weather):**
```python
import requests
import datetime

class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }
    
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        results = response.json()
        # Process and return current temperature
        return f'The current temperature is {current_temperature}°C'
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")
```

**Wikipedia Search Tool:**
```python
import wikipedia

@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[:3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except:
            pass
    return "\n\n".join(summaries) if summaries else "No good Wikipedia Search Result was found"
```

**Key Learning Points:**
- Tools provide structured interfaces to external systems
- Good descriptions help LLMs choose the right tool
- Type hints and schemas prevent errors
- Tools can be composed and chained
- Error handling is critical for reliability

### Lesson 6: Conversational Agents

**Objective:** Build conversational agents that maintain context and intelligently use multiple tools.

**Topics Covered:**
- Creating multi-tool agents
- Implementing agent memory and scratchpad
- Building conversational loops
- Agent output parsing
- Tool selection and routing
- Managing conversation history

**Setting Up Tools:**
```python
from langchain.tools import tool

# Define multiple tools
tools = [get_current_temperature, search_wikipedia]

# Format for OpenAI
from langchain.tools.render import format_tool_to_openai_function
functions = [format_tool_to_openai_function(f) for f in tools]
```

**Building the Agent:**
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

model = ChatOpenAI(temperature=0).bind(functions=functions)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

chain = prompt | model | OpenAIFunctionsAgentOutputParser()
```

**Agent Execution:**
```python
# Agent analyzes the query and selects appropriate tool
result = chain.invoke({
    "input": "what is the weather in sf?",
    "agent_scratchpad": []
})

# Result contains:
result.tool         # "get_current_temperature"
result.tool_input   # {"latitude": 37.7749, "longitude": -122.4194}
```

**Agent Scratchpad (Memory):**
The `agent_scratchpad` parameter stores:
- Previous tool calls and results
- Conversation context
- Intermediate reasoning steps
- Tool execution history

This enables the agent to:
- Remember previous interactions
- Build on prior tool results
- Maintain conversation continuity
- Make context-aware decisions

**Key Features:**
- **Tool Selection:** Agent chooses appropriate tool based on query
- **Context Awareness:** Remembers conversation history
- **Error Recovery:** Handles tool failures gracefully
- **Multi-Step Reasoning:** Chains multiple tool calls
- **Natural Responses:** Integrates tool results into conversational replies

## Key Concepts

### Function Calling vs. Traditional Prompting

| Aspect | Traditional Prompting | Function Calling |
|--------|---------------------|------------------|
| **Output** | Unstructured text | Structured JSON |
| **Validation** | Manual parsing required | Automatic type validation |
| **Reliability** | Prone to format variations | Consistent structure |
| **Integration** | String parsing needed | Direct object mapping |
| **Type Safety** | None | Full type checking |

### Pydantic Benefits

✅ **Key Advantages:**
- Automatic data validation
- Type safety at runtime
- Self-documenting schemas
- IDE autocomplete support
- Easy serialization/deserialization
- Nested model support

### Tool Design Principles

1. **Clear Descriptions:** Help LLMs understand tool purpose
2. **Type Hints:** Use proper Python type annotations
3. **Field Descriptions:** Document all parameters
4. **Error Handling:** Return meaningful error messages
5. **Single Responsibility:** One tool, one clear purpose
6. **Testability:** Easy to test in isolation

### When to Use Function Calling

✅ **Use function calling when:**
- You need structured data output
- Integrating with APIs or databases
- Building multi-step workflows
- Type safety is important
- Validating LLM outputs
- Creating tool-using agents

❌ **Don't use function calling when:**
- Simple text generation is sufficient
- No need for structured output
- One-off creative tasks
- Exploratory conversations

## Technologies Used

### Core Libraries

- **LangChain:** Framework for building LLM applications
- **OpenAI:** GPT models with function calling capability
- **Pydantic:** Data validation using Python type hints
- **Python-dotenv:** Environment variable management

### External APIs

- **Open-Meteo API:** Real-time weather data
- **Wikipedia API:** Encyclopedia search and summaries

### Key LangChain Components

- `ChatOpenAI`: OpenAI chat model wrapper
- `ChatPromptTemplate`: Prompt management
- `@tool` decorator: Tool creation
- `OpenAIFunctionsAgentOutputParser`: Parse agent outputs
- `JsonOutputFunctionsParser`: Convert function outputs to JSON
- `MessagesPlaceholder`: Manage conversation history

## Use Cases

### 1. Data Extraction Pipeline
Extract structured information from unstructured documents:
- Resume parsing
- Invoice processing
- Contract analysis
- Medical record extraction

### 2. Intelligent Customer Support
Build agents that can:
- Look up order status
- Check inventory
- Process returns
- Search knowledge bases

### 3. Research Assistant
Create agents that:
- Search multiple sources (Wikipedia, papers, docs)
- Aggregate information
- Fact-check claims
- Generate summaries

### 4. Content Tagging System
Automatically tag and categorize:
- Social media posts
- Product reviews
- Support tickets
- News articles

### 5. Multi-Tool Orchestration
Coordinate multiple services:
- Weather + Calendar (suggest meeting times)
- Maps + Restaurants (find nearby dining)
- Finance + News (investment research)

## Best Practices

### Pydantic Model Design

1. **Use Descriptive Docstrings**
```python
class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")
```

2. **Field Descriptions**
```python
class Person(BaseModel):
    name: str = Field(description="person's full name")
    age: Optional[int] = Field(description="person's age in years, if known")
    email: str = Field(description="person's email address")
```

3. **Use Optional for Nullable Fields**
```python
from typing import Optional

class UserProfile(BaseModel):
    name: str  # Required
    bio: Optional[str] = None  # Optional with default
```

### Tool Development

1. **Clear Tool Names**
```python
# Good
@tool
def get_current_temperature(latitude: float, longitude: float) -> str:
    """Fetch current temperature for given coordinates."""
    
# Bad
@tool
def tool1(x: float, y: float) -> str:
    """Does something."""
```

2. **Comprehensive Error Handling**
```python
@tool
def api_call(param: str) -> str:
    """Call external API with parameter."""
    try:
        response = requests.get(API_URL, params={"q": param})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return f"Error calling API: {str(e)}"
```

3. **Input Validation**
```python
class CoordinateInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")
```

### Agent Design

1. **System Prompts Matter**
```python
# Specific and directive
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when appropriate. Be concise."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
```

2. **Manage Context Window**
- Limit conversation history length
- Summarize old messages
- Clear scratchpad periodically

3. **Temperature Settings**
```python
# For structured tasks: low temperature
model = ChatOpenAI(temperature=0)

# For creative tasks: higher temperature
model = ChatOpenAI(temperature=0.7)
```

### Testing

1. **Test Tools Independently**
```python
# Test before integrating with LLM
def test_weather_tool():
    result = get_current_temperature(latitude=37.7749, longitude=-122.4194)
    assert "temperature" in result.lower()
```

2. **Validate Schemas**
```python
# Ensure Pydantic models validate correctly
def test_person_validation():
    valid_person = Person(name="John", age=30)
    assert valid_person.age == 30
    
    with pytest.raises(ValidationError):
        invalid_person = Person(name="Jane", age="thirty")
```

3. **Mock External APIs**
```python
from unittest.mock import patch

@patch('requests.get')
def test_weather_api(mock_get):
    mock_get.return_value.json.return_value = {"temperature": 20}
    result = get_current_temperature(0, 0)
    assert "20" in result
```

## Environment Setup

### Create `.env` file

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install langchain langchain-openai openai pydantic python-dotenv requests wikipedia

# For Jupyter notebooks
pip install jupyter ipython
jupyter notebook
```

### Basic Setup Code

```python
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())

# Verify API key is loaded
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
```

## Troubleshooting

### Common Issues

**Problem:** ValidationError when creating Pydantic models
- **Solution:** Check that all required fields are provided and types match

**Problem:** Function not being called by LLM
- **Solution:** Improve function description, ensure parameters are well-documented

**Problem:** API rate limits or timeout errors
- **Solution:** Implement retry logic, add delays between calls, check API quotas

**Problem:** Agent not using tools appropriately
- **Solution:** Refine system prompt, improve tool descriptions, adjust temperature

**Problem:** ImportError for langchain modules
- **Solution:** Update to latest langchain version: `pip install --upgrade langchain`

### Debugging Tips

1. **Print Function Definitions**
```python
from langchain.tools.render import format_tool_to_openai_function
print(format_tool_to_openai_function(your_tool))
```

2. **Inspect Agent Scratchpad**
```python
result = chain.invoke({
    "input": "query",
    "agent_scratchpad": previous_messages
})
print(result)  # See what the agent decided
```

3. **Test Pydantic Validation**
```python
try:
    model_instance = YourModel(**data)
except ValidationError as e:
    print(e.json())  # Detailed error information
```

## Advanced Topics

### Custom Output Parsers

Create parsers for specific output formats:
```python
from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=YourModel)
```

### Tool with Callbacks

Monitor tool execution:
```python
from langchain.callbacks import StdOutCallbackHandler

chain.invoke(
    {"input": "query"},
    config={"callbacks": [StdOutCallbackHandler()]}
)
```

### Multi-Agent Systems

Combine multiple agents with different specializations:
- Research agent (Wikipedia, search)
- Data agent (APIs, databases)
- Analysis agent (processing, summarization)

### Streaming Responses

Stream agent responses for better UX:
```python
for chunk in chain.stream({"input": "query"}):
    print(chunk, end="", flush=True)
```

## Next Steps

1. **Build Your Own Tools:** Create tools for your specific APIs and services
2. **Advanced Agents:** Explore ReAct, Plan-and-Execute agent patterns
3. **Memory Systems:** Implement conversation memory and retrieval
4. **Production Deployment:** Add monitoring, error handling, and scaling
5. **LangSmith:** Use LangSmith for debugging and monitoring agents

## Additional Resources

### Documentation
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)

### Related Courses
- [DeepLearning.AI - LangChain Courses](https://learn.deeplearning.ai/)
- [LangChain Agents Deep Dive](https://learn.deeplearning.ai/courses/langchain-chat-with-your-data)

### Community
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Discord](https://discord.gg/langchain)
- [OpenAI Community Forum](https://community.openai.com/)

## License

This educational material follows the licenses of the underlying frameworks:
- LangChain: MIT License
- OpenAI API: Commercial license (API usage)
- Pydantic: MIT License

## Acknowledgments

- DeepLearning.AI for the comprehensive course
- LangChain team for the excellent framework
- OpenAI for function calling capabilities
- Open-Meteo for free weather API access

---

**Note:** These are student notebooks from an educational course. For production use, ensure proper error handling, security measures, API key management, and testing.