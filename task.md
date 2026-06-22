# CloudDash Multi-Agent Customer Support System

## What is the Project
CloudDash is a B2B SaaS platform that provides real-time monitoring, alerting, and cost optimization for cloud infrastructure. This project involves building a prototype multi-agent customer support system that can handle end-to-end customer interactions for this fictional platform.

## What We Are Doing in the Project
We are implementing an orchestrator that routes user queries to specialized AI agents based on intent. The system will:
* Use a RAG pipeline to ground technical answers in a provided knowledge base.
* Query a local SQL database to handle billing inquiries and simulated account changes.
* Seamlessly hand over context between agents when a conversation crosses domain boundaries.
* Escalate to a simulated human operator when the AI cannot resolve the issue.
* Run behind a simple REST API.

## How the Project Should Work
1. A user sends a message via the REST API.
2. The **Triage Agent** intercepts the message, classifies the intent, and extracts initial context.
3. The query is routed to the appropriate specialist:
   * **Technical Support Agent**: Uses FAISS vector search to retrieve troubleshooting steps and API docs.
   * **Billing Agent**: Uses local SQL to look up account details and process simulated plan changes.
4. If the user asks a follow-up question requiring a different domain, the active agent hands the conversation over to the correct agent, preserving full history and extracted entities.
5. If an issue is beyond the AI's capabilities or requires human authority (e.g., refunds), the system routes to the **Escalation Agent** to package the context and simulate human handover.
6. Every step is logged with a unique trace ID, and inputs/outputs are checked against safety guardrails.

## Tech Stack
* **Orchestration**: LangChain & LangGraph
* **LLM**: Google Gemini
* **Embeddings**: `all-MiniLM-L6-v2`
* **Vector Store**: FAISS
* **Relational Database**: Local SQLite (for Billing Data)
* **Backend Framework**: FastAPI
* **Data Validation**: Pydantic

---

## Implementation Stages

Follow these stages sequentially. Do not move to the next stage until the current stage is fully implemented and tested. Use the designated IDE skills for optimized code generation.

### Stage 1: Core Setup & Data Models
**IDE Skill Trigger**: `@langchain-architecture`
* **Task**: Initialize the Python environment and define the core structures.
* **Requirements**:
  * Set up `requirements.txt` (langchain, langgraph, faiss-cpu, sentence-transformers, google-generativeai, fastapi, uvicorn, pydantic).
  * Create Pydantic models for Conversation State, Agent Responses, and Handover Payloads.
  * Implement a YAML configuration loader for agent system prompts and routing rules so they are not hardcoded.

### Stage 2: Knowledge Base & RAG Pipeline
**IDE Skill Trigger**: `@rag-engineer`
* **Task**: Build the ingestion and retrieval engine for the Technical Support Agent.
* **Requirements**:
  * Create a script to generate 15-20 sample Knowledge Base articles in JSON format covering FAQs, Troubleshooting, Billing, and API Docs.
  * Implement document chunking and embedding using `all-MiniLM-L6-v2`.
  * Initialize a FAISS vector store and index the chunks.
  * Create a retrieval chain that rewrites user queries using conversation context before searching.
  * Ensure the retrieval output includes source document IDs for citation.

### Stage 3: Local SQL & Tool Definition
**IDE Skill Trigger**: `@langchain-architecture`
* **Task**: Create the mock database and LangChain tools for the agents.
* **Requirements**:
  * Set up a local SQLite database to represent CloudDash users, subscriptions, and invoices.
  * Populate the database with dummy data.
  * Create LangChain tools (using `@tool` decorator) for:
    * `lookup_account_billing_info(customer_id)`
    * `process_plan_upgrade(customer_id, new_plan)`
    * `search_technical_knowledge_base(query)` (connecting to Stage 2's FAISS index).

### Stage 4: Agent Node Implementations
**IDE Skill Trigger**: `@langchain-architecture`
* **Task**: Define the distinct agents using Gemini and the initialized tools.
* **Requirements**:
  * **Triage Agent**: Prompted strictly for intent classification (Technical, Billing, General, Escalation) and entity extraction.
  * **Technical Support Agent**: Prompted to use the FAISS retrieval tool and format step-by-step answers with strict source citations.
  * **Billing Agent**: Prompted to use SQL tools for account lookup and strictly adhere to billing policy.
  * **Escalation Agent**: Prompted to summarize the conversation, classify priority, and output a simulated handover package.

### Stage 5: LangGraph Orchestration & Handover
**IDE Skill Trigger**: `@langgraph`
* **Task**: Connect the agents into a stateful graph network.
* **Requirements**:
  * Define the `GraphState` (TypedDict) containing `messages`, `current_agent`, `customer_id`, and `trace_id`.
  * Implement the node functions calling the agents built in Stage 4.
  * Define conditional routing edges based on the output of the Triage Agent or the active agent requesting a handover.
  * Implement the handover protocol ensuring history is preserved and allowing graceful fallback to Triage if an agent fails.
  * Compile the graph.

### Stage 6: Guardrails & Observability
**IDE Skill Trigger**: `@langchain-architecture`
* **Task**: Ensure the system is safe, stable, and transparent.
* **Requirements**:
  * Implement an input guardrail (e.g., prompt injection detection) before the query hits the graph.
  * Implement an output guardrail to prevent hallucination of CloudDash pricing/policies.
  * Integrate structured JSON logging across the application, tagging every agent invocation and handover event with a unique session `trace_id`.

### Stage 7: REST API Interface
**IDE Skill Trigger**: `@langchain-architecture`
* **Task**: Expose the LangGraph workflow via FastAPI.
* **Requirements**:
  * Build a `/chat` endpoint that accepts a user message and a session ID.
  * Build a `/history` endpoint to retrieve the conversation state for a given session ID.
  * Structure the application cleanly, keeping API routing separate from agent logic.