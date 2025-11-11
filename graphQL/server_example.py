"""
GraphQL Server Example using Strawberry
Run with: python server_example.py
Access GraphQL Playground at: http://localhost:8000/graphql
"""
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
app = FastAPI(title="GraphQL Example Server")
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

if __name__ == "__main__":
    import uvicorn
    print("Starting GraphQL server...")
    print("GraphQL endpoint: http://localhost:8000/graphql")
    print("GraphQL Playground: http://localhost:8000/graphql")
    uvicorn.run(app, host="0.0.0.0", port=8000)

