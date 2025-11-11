"""
GraphQL Client Example using gql
Make sure the server is running before executing this script.
Run server with: python server_example.py
Then run this client: python client_example.py
"""
from gql import gql, Client
from gql.transport.httpx import HTTPXTransport
from typing import Dict, Any, Optional

# Create a transport
transport = HTTPXTransport(url="http://localhost:8000/graphql")

# Create a GraphQL client
client = Client(transport=transport, fetch_schema_from_transport=True)

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

def update_user(user_id: int, name: Optional[str] = None, 
                email: Optional[str] = None) -> Dict[str, Any]:
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
    import json
    
    print("=" * 60)
    print("GraphQL Client Examples")
    print("=" * 60)
    
    # Test queries
    print("\n1. Get User by ID:")
    print(json.dumps(get_user(1), indent=2))
    
    print("\n2. Get All Users:")
    print(json.dumps(get_all_users(), indent=2))
    
    print("\n3. Get User with Posts:")
    print(json.dumps(get_user_with_posts(1), indent=2))
    
    print("\n4. Create New User:")
    print(json.dumps(create_user("Charlie", "charlie@example.com"), indent=2))
    
    print("\n5. Update User:")
    print(json.dumps(update_user(1, name="Alice Updated"), indent=2))
    
    print("\n6. Get All Users (after changes):")
    print(json.dumps(get_all_users(), indent=2))

