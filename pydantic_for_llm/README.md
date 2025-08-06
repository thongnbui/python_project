# Pydantic for LLM Tutorial Series

This repository contains a comprehensive tutorial series on using Pydantic for building robust, type-safe LLM applications. The project demonstrates how to leverage Pydantic's data validation capabilities to create reliable, structured outputs from Large Language Models.

## 📁 Project Structure

```
pydantic_for_llm/
├── README.md                           # This file
├── lesson_2_basic_model.ipynb         # Lesson 2: Pydantic Basics (25KB, 714 lines)
├── lesson_4_validate_llm_response.ipynb # Lesson 4: Structured LLM Output (14KB, 514 lines)
└── lesson_5_tool_call.ipynb          # Lesson 5: Tool Calling with Pydantic (32KB, 973 lines)
```

## 🎯 Tutorial Overview

This tutorial series progresses from basic Pydantic concepts to advanced LLM integration patterns:

### Lesson 2: Pydantic Basics
**File**: `lesson_2_basic_model.ipynb`

- **Objective**: Master fundamental Pydantic data validation
- **Key Concepts**:
  - Creating Pydantic models for data validation
  - Handling validation errors gracefully
  - Working with optional fields and constraints
  - JSON data validation methods
- **Use Case**: Customer support system with user input validation
- **Technologies**: Pydantic BaseModel, ValidationError, EmailStr

### Lesson 4: Validate LLM Response
**File**: `lesson_4_validate_llm_response.ipynb`

- **Objective**: Use Pydantic models for structured LLM output
- **Key Concepts**:
  - Direct Pydantic model integration with LLM APIs
  - Structured output generation across multiple providers
  - Field validation and constraints
  - Type-safe LLM responses
- **Use Case**: Customer query analysis with structured output
- **Technologies**: OpenAI, Anthropic, Instructor, Pydantic-AI

### Lesson 5: Tool Calling with Pydantic
**File**: `lesson_5_tool_call.ipynb`

- **Objective**: Build robust tool calling systems with Pydantic validation
- **Key Concepts**:
  - Pydantic models for tool schema definition
  - Tool argument validation and parsing
  - Multi-step LLM workflows
  - Error handling in tool calling
- **Use Case**: Customer support agent with tool orchestration
- **Technologies**: OpenAI Tool Calling, Pydantic-AI Agent

## 🛠️ Technologies Used

### Core Dependencies
- **Pydantic**: Data validation and settings management
- **OpenAI**: GPT-4o for LLM interactions
- **Anthropic**: Claude for alternative LLM provider
- **Instructor**: Structured output generation
- **Pydantic-AI**: Agent-based workflows

### Key Pydantic Components
- `BaseModel` - Core data validation
- `Field` - Field configuration and constraints
- `ValidationError` - Error handling
- `EmailStr` - Email validation
- `field_validator` - Custom validation logic

## 📊 Key Features

### 1. **Robust Data Validation**
- Type-safe data models with automatic validation
- Custom field validators for complex business logic
- Graceful error handling with detailed error messages
- JSON schema generation for API documentation

### 2. **Structured LLM Output**
- Direct Pydantic model integration with LLM APIs
- Type-safe responses across multiple providers
- Automatic validation of LLM-generated content
- Consistent data structures regardless of LLM provider

### 3. **Tool Calling Integration**
- Pydantic models for tool schema definition
- Validated tool arguments and return values
- Multi-step workflow orchestration
- Error handling in complex tool chains

### 4. **Customer Support Use Case**
- User input validation for support tickets
- Customer query analysis and categorization
- Order status checking and FAQ lookup
- Priority assessment and tagging

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8+ recommended
python --version

# Install core dependencies
pip install pydantic openai anthropic instructor pydantic-ai python-dotenv
```

### Environment Setup
1. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

2. Ensure you have access to the required LLM providers

### Running the Tutorials

#### Lesson 2: Pydantic Basics
```bash
jupyter notebook lesson_2_basic_model.ipynb
```

#### Lesson 4: Validate LLM Response
```bash
jupyter notebook lesson_4_validate_llm_response.ipynb
```

#### Lesson 5: Tool Calling with Pydantic
```bash
jupyter notebook lesson_5_tool_call.ipynb
```

## 📚 Learning Path

### Beginner → Advanced Progression

1. **Start with Pydantic Basics** (Lesson 2)
   - Understand data validation fundamentals
   - Learn error handling patterns
   - Master field constraints and validation

2. **Progress to LLM Integration** (Lesson 4)
   - Integrate Pydantic with LLM APIs
   - Generate structured outputs
   - Work with multiple LLM providers

3. **Advance to Tool Calling** (Lesson 5)
   - Build complex tool orchestration
   - Implement multi-step workflows
   - Master tool validation and error handling

## 🔧 Core Models

### UserInput Model
```python
class UserInput(BaseModel):
    name: str
    email: EmailStr
    query: str
    order_id: Optional[int] = Field(None, ge=10000, le=99999)
    purchase_date: Optional[date] = None
```

### CustomerQuery Model
```python
class CustomerQuery(UserInput):
    priority: str = Field(..., description="Priority level: low, medium, high")
    category: Literal['refund_request', 'information_request', 'other']
    is_complaint: bool
    tags: List[str]
```

### Tool Argument Models
```python
class CheckOrderStatusArgs(BaseModel):
    order_id: str = Field(..., description="Order ID in format ABC-12345")
    email: str = Field(..., description="Customer email address")

class FAQLookupArgs(BaseModel):
    query: str = Field(..., description="FAQ search query")
    category: Optional[str] = Field(None, description="FAQ category")
```

## 📈 Sample Use Cases

### 1. **Customer Support System**
- Validate user input for support tickets
- Categorize and prioritize customer queries
- Generate structured responses with proper validation

### 2. **Order Management**
- Validate order IDs with custom format checking
- Check order status with tool calling
- Handle order-related queries with structured data

### 3. **FAQ System**
- Search FAQ database with validated queries
- Return structured answers with metadata
- Integrate with customer support workflow

## 🎓 Key Learning Outcomes

After completing this tutorial series, you will be able to:

- ✅ Create robust Pydantic models for data validation
- ✅ Handle validation errors gracefully in LLM applications
- ✅ Generate structured outputs from multiple LLM providers
- ✅ Build type-safe tool calling systems
- ✅ Implement complex multi-step workflows
- ✅ Design reliable customer support systems

## 🔍 Advanced Concepts Covered

- **Data Validation**: Type checking, custom validators, field constraints
- **Error Handling**: Graceful validation error management
- **LLM Integration**: Direct model integration with OpenAI, Anthropic
- **Tool Calling**: Pydantic-based tool schema definition
- **Workflow Orchestration**: Multi-step agent workflows
- **Type Safety**: End-to-end type safety in LLM applications

## 🛡️ Best Practices

### 1. **Model Design**
- Use descriptive field names and descriptions
- Implement custom validators for business logic
- Leverage Pydantic's built-in validators when possible

### 2. **Error Handling**
- Catch ValidationError exceptions gracefully
- Provide meaningful error messages to users
- Log validation errors for debugging

### 3. **LLM Integration**
- Use structured output for consistent responses
- Validate LLM outputs before processing
- Handle malformed responses gracefully

### 4. **Tool Calling**
- Define clear tool schemas with Pydantic
- Validate tool arguments before execution
- Handle tool execution errors appropriately

## 🤝 Contributing

This tutorial series is designed for educational purposes. Feel free to:

- Extend the examples with your own use cases
- Add new validation patterns and models
- Experiment with different LLM providers
- Share improvements and enhancements

## 📄 License

This project is for educational purposes. Please ensure proper attribution when using any code or methodologies.

---

*Built with Pydantic and OpenAI - Type-Safe LLM Applications Tutorial Series* 