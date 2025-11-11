# GraphQL Guide for Python

## What is GraphQL?

GraphQL is a query language for APIs and a runtime for executing those queries. It was developed by Facebook in 2012 and open-sourced in 2015. GraphQL provides a more efficient, powerful, and flexible alternative to REST APIs.

### Key Concepts

1. **Single Endpoint**: Unlike REST which uses multiple endpoints, GraphQL uses a single endpoint (typically `/graphql`)
2. **Client-Driven Queries**: Clients specify exactly what data they need
3. **Strongly Typed Schema**: GraphQL APIs are built on a schema that defines all available data types
4. **Introspection**: GraphQL APIs are self-documenting and can be queried for their schema

### GraphQL vs REST

| Feature | REST | GraphQL |
|---------|------|---------|
| Endpoints | Multiple endpoints | Single endpoint |
| Data Fetching | Fixed data structure | Client specifies fields |
| Over-fetching | Common problem | Avoided |
| Under-fetching | May require multiple requests | Single request |
| Versioning | URL versioning (v1, v2) | Schema evolution |

## Core GraphQL Operations

### 1. Queries (Read Operations)
Fetch data from the server.

```graphql
query {
  user(id: "123") {
    name
    email
    posts {
      title
      content
    }
  }
}
```

### 2. Mutations (Write Operations)
Modify data on the server.

```graphql
mutation {
  createUser(name: "John", email: "john@example.com") {
    id
    name
    email
  }
}
```

### 3. Subscriptions (Real-time Updates)
Receive real-time updates from the server.

```graphql
subscription {
  messageAdded {
    id
    content
    author {
      name
    }
  }
}
```

## Python GraphQL Libraries

### Popular Libraries

1. **Strawberry GraphQL** - Modern, type-safe, async-first
2. **Graphene** - Mature, Django-friendly
3. **Ariadne** - Schema-first approach
4. **gql** - Client library for making GraphQL queries
5. **tartiflette** - Async GraphQL engine

## Example 1: Creating a GraphQL Server with Strawberry

### Installation

```bash
pip install strawberry[fastapi] uvicorn
```

### Server Implementation

```python
# server.py
from typing import List, Optional
import strawberry
from strawberry.fastapi import GraphQLRouter
from fastapi import FastAPI

# Define the data models
@strawberry.type
class User:
    id: int
    name: str
    email: str

@strawberry.type
class Post:
    id: int
    title: str
    content: str
    author_id: int

# In-memory database (for demo purposes)
users_db = [
    User(id=1, name="Alice", email="alice@example.com"),
    User(id=2, name="Bob", email="bob@example.com"),
]

posts_db = [
    Post(id=1, title="Hello World", content="My first post", author_id=1),
    Post(id=2, title="GraphQL is Great", content="Learning GraphQL", author_id=2),
]

# Define the Query type
@strawberry.type
class Query:
    @strawberry.field
    def user(self, id: int) -> Optional[User]:
        """Get a user by ID"""
        return next((u for u in users_db if u.id == id), None)
    
    @strawberry.field
    def users(self) -> List[User]:
        """Get all users"""
        return users_db
    
    @strawberry.field
    def posts(self, author_id: Optional[int] = None) -> List[Post]:
        """Get posts, optionally filtered by author_id"""
        if author_id:
            return [p for p in posts_db if p.author_id == author_id]
        return posts_db

# Define the Mutation type
@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_user(self, name: str, email: str) -> User:
        """Create a new user"""
        new_id = max([u.id for u in users_db], default=0) + 1
        new_user = User(id=new_id, name=name, email=email)
        users_db.append(new_user)
        return new_user
    
    @strawberry.mutation
    def update_user(self, id: int, name: Optional[str] = None, 
                    email: Optional[str] = None) -> Optional[User]:
        """Update an existing user"""
        user = next((u for u in users_db if u.id == id), None)
        if not user:
            return None
        
        if name:
            user.name = name
        if email:
            user.email = email
        
        return user

# Create the schema
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Create FastAPI app with GraphQL endpoint
app = FastAPI()
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Running the Server

```bash
python server.py
```

The GraphQL endpoint will be available at: `http://localhost:8000/graphql`

You can also access GraphQL Playground (interactive IDE) at: `http://localhost:8000/graphql`

## Example 2: Making GraphQL Queries from Python Client

### Installation

```bash
pip install gql[all] httpx
```

### Client Implementation

```python
# client.py
from gql import gql, Client
from gql.transport.httpx import HTTPXTransport
from typing import Dict, Any

# Create a transport
transport = HTTPXTransport(url="http://localhost:8000/graphql")

# Create a GraphQL client
client = Client(transport=transport, fetch_schema_from_transport=True)

# Example 1: Query a single user
def get_user(user_id: int) -> Dict[str, Any]:
    """Get a user by ID"""
    query = gql("""
        query GetUser($id: Int!) {
            user(id: $id) {
                id
                name
                email
            }
        }
    """)
    
    variables = {"id": user_id}
    result = client.execute(query, variable_values=variables)
    return result

# Example 2: Query all users
def get_all_users() -> Dict[str, Any]:
    """Get all users"""
    query = gql("""
        query GetAllUsers {
            users {
                id
                name
                email
            }
        }
    """)
    
    result = client.execute(query)
    return result

# Example 3: Query with nested fields
def get_user_with_posts(user_id: int) -> Dict[str, Any]:
    """Get user with their posts"""
    query = gql("""
        query GetUserWithPosts($id: Int!) {
            user(id: $id) {
                id
                name
                email
            }
            posts(authorId: $id) {
                id
                title
                content
            }
        }
    """)
    
    variables = {"id": user_id}
    result = client.execute(query, variable_values=variables)
    return result

# Example 4: Create a user (Mutation)
def create_user(name: str, email: str) -> Dict[str, Any]:
    """Create a new user"""
    mutation = gql("""
        mutation CreateUser($name: String!, $email: String!) {
            createUser(name: $name, email: $email) {
                id
                name
                email
            }
        }
    """)
    
    variables = {"name": name, "email": email}
    result = client.execute(mutation, variable_values=variables)
    return result

# Example 5: Update a user (Mutation)
def update_user(user_id: int, name: str = None, email: str = None) -> Dict[str, Any]:
    """Update an existing user"""
    mutation = gql("""
        mutation UpdateUser($id: Int!, $name: String, $email: String) {
            updateUser(id: $id, name: $name, email: $email) {
                id
                name
                email
            }
        }
    """)
    
    variables = {"id": user_id}
    if name:
        variables["name"] = name
    if email:
        variables["email"] = email
    
    result = client.execute(mutation, variable_values=variables)
    return result

if __name__ == "__main__":
    # Test queries
    print("=== Get User ===")
    print(get_user(1))
    
    print("\n=== Get All Users ===")
    print(get_all_users())
    
    print("\n=== Create User ===")
    print(create_user("Charlie", "charlie@example.com"))
    
    print("\n=== Update User ===")
    print(update_user(1, name="Alice Updated"))
```

## Example 3: Using GraphQL with Async/Await

```python
# async_client.py
import asyncio
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

async def async_query_example():
    """Example of async GraphQL queries"""
    transport = AIOHTTPTransport(url="http://localhost:8000/graphql")
    
    async with Client(transport=transport, fetch_schema_from_transport=True) as session:
        # Query
        query = gql("""
            query {
                users {
                    id
                    name
                    email
                }
            }
        """)
        
        result = await session.execute(query)
        return result

# Run async example
if __name__ == "__main__":
    result = asyncio.run(async_query_example())
    print(result)
```

## Example 4: Error Handling

```python
# error_handling.py
from gql import gql, Client
from gql.transport.httpx import HTTPXTransport
from gql.transport.exceptions import TransportQueryError

def safe_query_example():
    """Example with proper error handling"""
    transport = HTTPXTransport(url="http://localhost:8000/graphql")
    client = Client(transport=transport, fetch_schema_from_transport=True)
    
    query = gql("""
        query GetUser($id: Int!) {
            user(id: $id) {
                id
                name
                email
            }
        }
    """)
    
    try:
        result = client.execute(query, variable_values={"id": 999})
        return result
    except TransportQueryError as e:
        print(f"GraphQL Error: {e.errors}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

## Example 5: GraphQL with Authentication

```python
# authenticated_client.py
from gql import gql, Client
from gql.transport.httpx import HTTPXTransport

def authenticated_query_example(api_key: str):
    """Example with authentication headers"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    transport = HTTPXTransport(
        url="https://api.example.com/graphql",
        headers=headers
    )
    
    client = Client(transport=transport, fetch_schema_from_transport=True)
    
    query = gql("""
        query {
            me {
                id
                name
                email
            }
        }
    """)
    
    result = client.execute(query)
    return result
```

## Example 6: GraphQL Subscriptions (Real-time)

```python
# subscription_example.py
import asyncio
from gql import gql, Client
from gql.transport.websockets import WebsocketsTransport

async def subscription_example():
    """Example of GraphQL subscriptions"""
    transport = WebsocketsTransport(url="ws://localhost:8000/graphql")
    
    async with Client(transport=transport, fetch_schema_from_transport=True) as session:
        subscription = gql("""
            subscription {
                messageAdded {
                    id
                    content
                    author {
                        name
                    }
                }
            }
        """)
        
        async for result in session.subscribe(subscription):
            print(f"New message: {result}")

# Run subscription
if __name__ == "__main__":
    asyncio.run(subscription_example())
```

## GraphQL Schema Definition

GraphQL schemas are defined using the Schema Definition Language (SDL):

```graphql
type User {
  id: Int!
  name: String!
  email: String!
  posts: [Post!]
}

type Post {
  id: Int!
  title: String!
  content: String!
  author: User!
}

type Query {
  user(id: Int!): User
  users: [User!]!
  posts(authorId: Int): [Post!]!
}

type Mutation {
  createUser(name: String!, email: String!): User!
  updateUser(id: Int!, name: String, email: String): User
  deleteUser(id: Int!): Boolean!
}
```

### Type System

- `String`, `Int`, `Float`, `Boolean` - Scalar types
- `!` - Non-nullable (required)
- `[Type]` - List of Type
- `[Type!]!` - Non-nullable list of non-nullable items

## Best Practices

1. **Use Variables**: Always use variables instead of string interpolation
   ```python
   # Good
   query = gql("query GetUser($id: Int!) { user(id: $id) { name } }")
   client.execute(query, variable_values={"id": 1})
   
   # Bad
   query = gql(f"query { user(id: 1) { name } }")
   ```

2. **Handle Errors**: Always wrap GraphQL queries in try-except blocks

3. **Use Fragments**: Reuse common field selections
   ```graphql
   fragment UserFields on User {
     id
     name
     email
   }
   
   query {
     user(id: 1) {
       ...UserFields
     }
   }
   ```

4. **Request Only Needed Fields**: One of GraphQL's main advantages
   ```graphql
   # Only request what you need
   query {
     user(id: 1) {
       name  # Don't request email if you don't need it
     }
   }
   ```

5. **Use Introspection**: Query the schema to understand available types
   ```python
   introspection_query = gql("""
     query IntrospectionQuery {
       __schema {
         types {
           name
           fields {
             name
             type {
               name
             }
           }
         }
       }
     }
   """)
   ```

## Testing GraphQL APIs

```python
# test_graphql.py
import pytest
from gql import gql, Client
from gql.transport.httpx import HTTPXTransport

@pytest.fixture
def client():
    transport = HTTPXTransport(url="http://localhost:8000/graphql")
    return Client(transport=transport, fetch_schema_from_transport=True)

def test_get_user(client):
    query = gql("""
        query GetUser($id: Int!) {
            user(id: $id) {
                id
                name
            }
        }
    """)
    
    result = client.execute(query, variable_values={"id": 1})
    assert result["user"]["id"] == 1
    assert result["user"]["name"] == "Alice"
```

## Resources

- [GraphQL Official Documentation](https://graphql.org/)
- [Strawberry GraphQL Documentation](https://strawberry.rocks/)
- [Graphene Documentation](https://docs.graphene-python.org/)
- [gql Python Client](https://gql.readthedocs.io/)
- [GraphQL Best Practices](https://graphql.org/learn/best-practices/)

## Summary

GraphQL provides:
- ✅ **Efficient data fetching** - Get exactly what you need
- ✅ **Single endpoint** - No versioning headaches
- ✅ **Strong typing** - Catch errors early
- ✅ **Self-documenting** - Introspection built-in
- ✅ **Flexible queries** - Client controls the response shape

Python has excellent support for GraphQL with libraries like Strawberry, Graphene, and gql, making it easy to both build GraphQL servers and consume GraphQL APIs.

