# ------------------------------------------------------------
# Imports MCP (Model Context Protocol)
# ------------------------------------------------------------

# Gestion de sessions MCP et paramètres serveur (stdio)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Adaptateurs LangChain pour charger les outils MCP
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

# Agent ReAct basé sur LangGraph
from langgraph.prebuilt import create_react_agent

# Callbacks pour le logging et le tracing
from langchain_community.callbacks import get_openai_callback
from langchain_core.tracers import ConsoleCallbackHandler

# Librairies standards
import asyncio
import time
import os

# Chargement des variables d’environnement (.env)
from dotenv import load_dotenv
load_dotenv()

# Fonctions utilitaires définies dans un script précédent
# - get_model : initialise le LLM (ex: Ollama)
# - interpret_agent_response : reformule la réponse brute
from main_simple import get_model, interpret_agent_response


# ------------------------------------------------------------
# Configuration de plusieurs serveurs MCP
# ------------------------------------------------------------

# Chaque serveur MCP expose des outils différents
# (Cypher, data modeling, mémoire, etc.)
MCP_SERVER_CONFIGS = {
    "neo4j-cypher": {
        "command": "uvx",
        "args": ["mcp-neo4j-cypher@0.2.4", "--transport", "stdio"],
        "transport": "stdio",
        "env": os.environ
    },
    "neo4j-data-modeling": {
        "command": "uvx",
        "args": ["mcp-neo4j-data-modeling@0.1.1", "--transport", "stdio"],
        "transport": "stdio",
        "env": os.environ
    },
    "memory": {
        "command": "uvx",
        "args": ["mcp-neo4j-memory@0.1.5"],
        "transport": "stdio",
        "env": os.environ
    },
    # Exemple commenté : nécessite un compte Neo4j Aura payant
    # "neo4j-aura": {
    #     "command": "uvx",
    #     "args": ["mcp-neo4j-aura-manager@0.2.2"],
    #     "transport": "stdio",
    #     "env": os.environ
    # }
}


# ------------------------------------------------------------
# Méthode "moins élégante" : chargement outil par outil (stdio)
# ------------------------------------------------------------

async def get_tools_from_server(server_name: str, server_cfg: dict):
    """
    Initialise un serveur MCP via stdio et récupère ses outils.
    """
    params = StdioServerParameters(
        command=server_cfg["command"],
        args=server_cfg["args"],
        transport=server_cfg["transport"],
        env=server_cfg["env"]
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return tools


async def get_all_tools(configs: dict):
    """
    Récupère les outils de tous les serveurs MCP en parallèle.
    """
    all_tools_lists = await asyncio.gather(*[
        get_tools_from_server(name, cfg)
        for name, cfg in configs.items()
    ])

    # Aplatit la liste des listes
    all_tools = [tool for tools in all_tools_lists for tool in tools]

    # Affiche les outils disponibles
    print("\nOutils MCP disponibles :")
    for tool in all_tools:
        print(f"- {tool.name}: {tool.description}")

    return all_tools


# ------------------------------------------------------------
# Méthode "élégante" : MultiServerMCPClient
# ------------------------------------------------------------

async def get_multi_tools(configs: dict):
    """
    Utilise MultiServerMCPClient pour charger tous les outils
    (stdio, SSE, HTTP streamable).
    """
    client = MultiServerMCPClient(configs)
    return await client.get_tools()


# ------------------------------------------------------------
# Classe Agent multi-outils (LangGraph + MCP + LLM)
# ------------------------------------------------------------

class MultiToolAgent:
    def __init__(self, model: str, configs: dict):
        self.model = model
        self.configs = configs
        self.agent = None
        self.tools = None

    async def initialize(self):
        """
        Initialise l’agent avec tous les outils MCP.
        """
        # Chargement élégant des outils MCP
        self.tools = await get_multi_tools(self.configs)

        # Création de l’agent ReAct (raisonnement + actions)
        self.agent = create_react_agent(
            get_model(self.model),
            self.tools
        )
        return self

    async def run_request(self, request: str, with_logging: bool = False) -> dict:
        """
        Exécute une requête utilisateur avec ou sans logging détaillé.
        """
        if not self.agent:
            await self.initialize()

        start_time = time.time()

        if with_logging:
            print(f"\n{'='*50}\nRequête : {request}\n{'='*50}")
            callbacks = [ConsoleCallbackHandler()]

            # Cas spécifique OpenAI (comptage de tokens)
            if 'gpt' in self.model.lower():
                with get_openai_callback() as cb:
                    agent_response = await self.agent.ainvoke(
                        {"messages": request},
                        {"callbacks": callbacks}
                    )
                    print(f"\nUtilisation tokens : {cb}")
            else:
                agent_response = await self.agent.ainvoke(
                    {"messages": request},
                    {"callbacks": callbacks}
                )

            print(f"\nRéponse brute :\n{agent_response}")
            interpreted = await interpret_agent_response(
                agent_response,
                request,
                self.model
            )
            print(f"\nRéponse finale :\n{interpreted}")

        else:
            agent_response = await self.agent.ainvoke(
                {"messages": request}
            )
            interpreted = await interpret_agent_response(
                agent_response,
                request,
                self.model
            )

        return {
            "raw": agent_response,
            "answer": interpreted,
            "seconds_to_complete": round(time.time() - start_time, 2)
        }


# ------------------------------------------------------------
# Point d’entrée du script
# ------------------------------------------------------------

if __name__ == "__main__":

    # Modèle LLM local (via Ollama)
    model = "llama3.1"

    # Exemple écriture
    # request = "Create a new node with the label 'Person' and the property 'name' set to 'John Doe'."

    # Exemple lecture
    request = "How many nodes are in the graph?"

    async def main():
        print("Initialisation de l’agent...")
        agent = await MultiToolAgent(model, MCP_SERVER_CONFIGS).initialize()
        print("Traitement de la requête...")
        result = await agent.run_request(request)
        print(result)

    asyncio.run(main())
