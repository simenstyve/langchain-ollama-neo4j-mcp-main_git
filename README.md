# langchain-ollama-neo4j-mcp-main_git

Multi-Server Agentic Knowledge Graph with MCP & LangChain

Strategic Overview
This project demonstrates a production-ready AI Solution Delivery framework that bridges the gap between unstructured data and complex reasoning. By integrating Generative AI (LLMs) with Neo4j Knowledge Graphs via the Model Context Protocol (MCP), this system provides an auditable, "grounded" alternative to standard black-box AI.

For mission-driven organizations like Human Rights Watch, this architecture enables researchers to query vast webs of evidence, legal precedents, and field reports with high factual accuracy and traceability.

Key Capabilities

Agentic Reasoning: 
Uses LangGraph ReAct agents to autonomously decide when to query the knowledge graph or use internal memory tools.

Knowledge Grounding (RAG): 
Connects Ollama (Llama 3) to a Neo4j backend to ensure responses are grounded in structured facts, significantly reducing hallucinations.

Multi-Server Orchestration: 
Utilizes a MultiServerMCPClient to manage separate tools for Cypher query generation, data modeling, and persistent memory.

Human-Centric Interface: 
A FastAPI backend and Streamlit frontend allow non-technical stakeholders to interact with complex graph data visually and through natural language.

Technology Stack
Orchestration: LangChain & LangGraph
Protocol: Model Context Protocol (MCP)
LLM: Ollama (Local deployment for data privacy)
Database: Neo4j (Knowledge Graph)
API & UI: FastAPI, Streamlit, Pyvis (Graph Visualization)

Responsible AI & Governance
Aligned with HRW’s focus on Transparency and Accountability, this project implements several safety and governance features:
Traceability: Every agent action is logged. Users can see the exact Cypher query used to retrieve data, ensuring the "Chain of Thought" is auditable.
Data Sovereignty: By utilizing Ollama for local LLM execution, sensitive organizational data remains within secure boundaries rather than being sent to third-party APIs.
Interpretable Outputs: Includes an interpretation layer that translates raw graph data into human-readable summaries without losing the underlying context.

Getting Started

1. Prerequisites
Neo4j Instance: Running local or Aura instance with APOC plugin.
Ollama: Installed and running with llama3.1 or llama3.2.
Python 3.10+

3. Configuration
Create a .env file in the root directory:


Code snippet
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000


3. Installation
   
Bash
pip install -r requirements.txt

5. Running the Solution
Start the Backend (FastAPI):
Bash
python main_fastapi.py
Start the Frontend (Streamlit):
Bash
streamlit run main_streamlit.py
