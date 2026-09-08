# Knowledge Graphs for RAG

Hands-on lessons for building and querying knowledge graphs with **Neo4j**, **Cypher**, **LangChain**, and **OpenAI**, then using them to power retrieval-augmented generation (RAG).

You progress from basic Cypher on a movie graph, through vector search and SEC document graphs, to natural-language Q&A via LLM-generated Cypher.

## When to Use a Knowledge Graph (and When Not To)

Use a knowledge graph when **relationships are the point of the query** — not just similar text.

### Use one when

- You need **multi-hop answers**: e.g. “managers near Santa Clara who own NetApp,” consecutive Form 10-K chunks, or investor → company → filing paths (as in L5–L7)
- Entities are **shared and reused** across documents (companies, people, filings) and you want one canonical node instead of duplicated mentions
- Retrieval should **follow structure** (`NEXT` windows, `PART_OF`, `OWNS_STOCK_IN`), not only embedding similarity
- You care about **explainability**: you can show the path/nodes that justified the answer
- Data is naturally a **network** (ownership, org charts, supply chains, citations, dependencies)

### Skip it when

- The job is **flat semantic search** (“find similar paragraphs”) — a vector store alone is simpler and enough (L3-style search without graph edges)
- Documents are **one-off** with little entity reuse; building nodes/edges costs more than it returns
- Requirements are mostly **keyword/filter lookup** on a table — SQL or a document DB is clearer
- You can’t maintain **entity resolution** (same company under many names) — a messy graph is worse than no graph
- Latency/ops budget is tight and you don’t need joins across relationships — Neo4j + embeddings + LLM Cypher is heavier than plain RAG

### Rule of thumb (from this course)

| Approach | Best for |
|----------|----------|
| **Vectors** | “What text is about this?” |
| **Graph** | “How do these things connect?” |
| **Both** | Answers that need relevant text *and* structured links (Form 10-K chunks + Form 13 ownership in L6–L7) |

If your questions never need “related to / owns / next to / filed by,” skip the KG and stay with chunked RAG.

## Project Structure

```
knowledge_graphs/
├── README.md                         # This file
├── L2-query_with_cypher.ipynb        # Cypher basics on a movie KG
├── L3-prep_text_for_RAG.ipynb        # Vector indexes & similarity search
├── L4-construct_kg_from_text.ipynb   # Build a KG from Form 10-K text
├── L5-add_relationships_to_kg.ipynb  # Chunk links & window retrieval
├── L6-expand_the_kg.ipynb            # Form 13s, companies & managers
└── L7-chat_with_kg.ipynb             # Chat with the graph (GraphCypherQAChain)
```

> **Note:** Lesson notebooks start at L2 (as provided). Data files such as `./data/form10k/` and `./data/form13.csv` are expected by later lessons when you run them in the full course environment.

## Prerequisites

- Python 3.10+
- A Neo4j instance (Aura or local) with vector index support
- An OpenAI API key
- Environment variables in a `.env` file:

```bash
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
OPENAI_API_KEY=sk-...
```

Typical packages used across the notebooks:

```bash
pip install python-dotenv langchain langchain-community langchain-openai neo4j openai
```

## Lesson Overview

### Lesson 2: Querying Knowledge Graphs with Cypher
**Notebook:** `L2-query_with_cypher.ipynb`

- Connect to Neo4j with LangChain’s `Neo4jGraph`
- Match nodes by label (`Movie`, `Person`) and properties
- Write Cypher patterns with conditions and multi-node paths
- Create and delete graph data

**Focus:** Cypher fundamentals on a movie knowledge graph.

---

### Lesson 3: Preparing Text Data for RAG
**Notebook:** `L3-prep_text_for_RAG.ipynb`

- Create a Neo4j **vector index** on movie taglines
- Embed taglines with OpenAI and store them on `Movie` nodes
- Run similarity search from a natural-language question

**Focus:** Bridging unstructured text embeddings with a graph database.

---

### Lesson 4: Constructing a Knowledge Graph from Text Documents
**Notebook:** `L4-construct_kg_from_text.ipynb`

- Load and inspect a company Form 10-K (NetApp) from SEC EDGAR-style JSON
- Chunk sections with LangChain’s `RecursiveCharacterTextSplitter`
- Create `Chunk` nodes, uniqueness constraints, and a vector index
- Embed chunks and run similarity search
- Build a `RetrievalQAWithSourcesChain` RAG workflow over the form

**Focus:** Turning long SEC filings into an embeddable chunk graph you can chat with.

---

### Lesson 5: Adding Relationships to the SEC Knowledge Graph
**Notebook:** `L5-add_relationships_to_kg.ipynb`

- Create a parent Form 10-K node and link chunks with `PART_OF` / `SECTION`
- Link ordered chunks with `NEXT` (including APOC `apoc.nodes.link`)
- Query paths and variable-length windows of consecutive chunks
- Customize vector retrieval with Cypher post-filters
- Compare single-chunk vs window-based RAG answers

**Focus:** Using graph structure (not just embeddings) to improve retrieval context.

---

### Lesson 6: Expanding the SEC Knowledge Graph
**Notebook:** `L6-expand_the_kg.ipynb`

- Load Form 13 filings for managers invested in NetApp
- Create `Company` and `Manager` nodes, full-text indexes, and `OWNS_STOCK_IN` / `FILED` relationships
- Walk paths from Form 10-K chunks → companies → investors
- Augment vector RAG with Cypher-derived investor context sentences

**Focus:** Enriching the document graph with real-world ownership relationships.

---

### Lesson 7: Chatting with the SEC Knowledge Graph
**Notebook:** `L7-chat_with_kg.ipynb`

- Explore schema, addresses, and geospatial-style queries (states, cities, nearby firms)
- Use full-text search (including typo-tolerant lookups)
- Teach an LLM to write Cypher with few-shot prompts via `GraphCypherQAChain`
- Extend the Cypher-generation prompt to pull Form 10-K chunk text into answers

**Focus:** End-to-end natural-language Q&A over a structured + unstructured SEC knowledge graph.

## Learning Path

```text
L2  Cypher basics (movies)
 └─► L3  Vector search on graph properties
      └─► L4  Chunk Form 10-K → nodes + RAG
           └─► L5  Relationships & window retrieval
                └─► L6  Form 13 investors & context injection
                     └─► L7  LLM → Cypher chat over the full KG
```

## Technologies

| Area | Tools |
|------|--------|
| Graph DB | Neo4j, Cypher, APOC |
| LLM / embeddings | OpenAI (Chat + embeddings) |
| Orchestration | LangChain (`Neo4jGraph`, `Neo4jVector`, `RetrievalQAWithSourcesChain`, `GraphCypherQAChain`) |
| Data | SEC Form 10-K JSON, Form 13 CSV |

## Suggested Workflow

1. Complete lessons in order (L2 → L7); later notebooks assume earlier graph concepts.
2. Keep the same Neo4j database across L4–L7 so relationships accumulate as intended.
3. After changing a Cypher retrieval query or prompt template, re-instantiate the vector store / QA chain before asking new questions.
4. Use the “Try for yourself” cells to experiment with questions, window sizes, and few-shot Cypher examples.

## Resources

- [Neo4j Cypher documentation](https://neo4j.com/product/cypher-graph-query-language/)
- [SEC EDGAR search](https://www.sec.gov/edgar/search/)
- [LangChain Neo4j integrations](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/)
