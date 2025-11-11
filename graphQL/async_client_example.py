"""
Async GraphQL Client Example
Demonstrates how to use GraphQL with async/await in Python
"""
import asyncio
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from typing import Dict, Any

async def async_get_users() -> Dict[str, Any]:
    """Example of async GraphQL query"""
    transport = AIOHTTPTransport(url="http://localhost:8000/graphql")
    
    async with Client(transport=transport, fetch_schema_from_transport=True) as session:
        query = gql("""
            query GetAllUsers {
                users {
                    id
                    name
                    email
                }
            }
        """)
        
        result = await session.execute(query)
        return result

async def async_create_user(name: str, email: str) -> Dict[str, Any]:
    """Example of async GraphQL mutation"""
    transport = AIOHTTPTransport(url="http://localhost:8000/graphql")
    
    async with Client(transport=transport, fetch_schema_from_transport=True) as session:
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
        result = await session.execute(mutation, variable_values=variables)
        return result

async def multiple_queries_example():
    """Example of running multiple queries concurrently"""
    transport = AIOHTTPTransport(url="http://localhost:8000/graphql")
    
    async with Client(transport=transport, fetch_schema_from_transport=True) as session:
        query1 = gql("""
            query {
                user(id: 1) {
                    id
                    name
                }
            }
        """)
        
        query2 = gql("""
            query {
                user(id: 2) {
                    id
                    name
                }
            }
        """)
        
        # Run queries concurrently
        results = await asyncio.gather(
            session.execute(query1),
            session.execute(query2)
        )
        
        return results

if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("Async GraphQL Client Examples")
    print("=" * 60)
    
    # Run async examples
    print("\n1. Get All Users (async):")
    result = asyncio.run(async_get_users())
    print(json.dumps(result, indent=2))
    
    print("\n2. Create User (async):")
    result = asyncio.run(async_create_user("David", "david@example.com"))
    print(json.dumps(result, indent=2))
    
    print("\n3. Multiple Queries Concurrently:")
    results = asyncio.run(multiple_queries_example())
    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}:")
        print(json.dumps(result, indent=2))

