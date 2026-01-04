# ------------------------------------------------------------
# Imports MCP, LangChain, LangGraph et Ollama
# ------------------------------------------------------------

# Client MCP pour communiquer avec un serveur via stdio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Outils MCP utilisables par LangChain
from langchain_mcp_adapters.tools import load_mcp_tools

# Agent ReAct préconstruit (Reason + Act)
from langgraph.prebuilt import create_react_agent

# Modèle LLM local via Ollama
from langchain_ollama import ChatOllama

# Librairies standard
import asyncio
import os

# Chargement des variables d’environnement depuis un fichier .env
from dotenv import load_dotenv
load_dotenv()


# ------------------------------------------------------------
# Configuration du serveur MCP (connexion via stdio)
# ------------------------------------------------------------

# Paramètres du serveur MCP utilisant Neo4j + Cypher
# La communication se fait via l'entrée/sortie standard (stdio)
server_params = StdioServerParameters(
    command="uvx",
    args=["mcp-neo4j-cypher@0.2.4", "--transport", "stdio"],
    transport="stdio",
    env=os.environ
)


# ------------------------------------------------------------
# Fonction utilitaire pour créer un modèle LLM Ollama
# ------------------------------------------------------------

def get_model(model_name):
    """
    Crée et retourne un modèle LLM via Ollama.
    """
    return ChatOllama(
        model=model_name,
        temperature=0.0,   # Température basse pour des réponses déterministes
        streaming=False   # Désactivation du streaming pour compatibilité
    )


# ------------------------------------------------------------
# Fonction utilitaire pour extraire le contenu texte d'une réponse LLM
# ------------------------------------------------------------

def extract_content(val):
    """
    Retourne le contenu texte d'une réponse LLM si disponible,
    sinon retourne la représentation string.
    """
    return val.content if hasattr(val, "content") else str(val)


# ------------------------------------------------------------
# Interprétation finale de la réponse brute de l'agent
# ------------------------------------------------------------

async def interpret_agent_response(agent_response, request, model_name="llama3.1"):
    """
    Utilise un LLM pour reformuler et interpréter la réponse brute
    générée par l'agent et les outils.
    """

    # Initialisation du modèle LLM
    llm = get_model(model_name)

    # Prompt d'interprétation
    prompt = (
        "You are an expert assistant. Given the following user request and the raw agent/tool response, "
        "return the most appropriate response to answer the user request.\n"
        f"User request: {request}\n"
        f"Agent/tool response: {agent_response}\n"
        "Answer:"
    )

    # Appel asynchrone ou synchrone selon le modèle
    if hasattr(llm, "ainvoke"):
        result = await llm.ainvoke(prompt)
    else:
        result = llm.invoke(prompt)

    # Extraction du texte final
    return extract_content(result)


# ------------------------------------------------------------
# Fonction principale : exécution de l'agent MCP + LLM
# ------------------------------------------------------------

async def run_agent(request: str, model: str) -> dict:
    """
    Exécute un agent LangGraph avec des outils MCP
    et retourne la réponse brute et interprétée.
    """

    # Connexion au serveur MCP via stdio
    async with stdio_client(server_params) as (read, write):

        # Création d'une session MCP
        async with ClientSession(read, write) as session:

            # Initialisation de la session
            await session.initialize()

            # Chargement des outils MCP disponibles (ex: Neo4j Cypher)
            tools = await load_mcp_tools(session)

            # Création de l'agent ReAct (raisonnement + actions)
            agent = create_react_agent(get_model(model), tools)

            # Envoi de la requête utilisateur à l'agent
            agent_response = await agent.ainvoke({"messages": request})

            # Interprétation finale de la réponse
            interpreted = await interpret_agent_response(
                agent_response,
                request,
                model
            )

            # Retour des deux versions de la réponse
            return {
                "raw": agent_response,
                "answer": interpreted
            }


# ------------------------------------------------------------
# Point d'entrée du script
# ------------------------------------------------------------

if __name__ == "__main__":

    # Nom du modèle Ollama à utiliser
    # Exécuter `ollama list` pour voir les modèles disponibles
    model = "llama3.1"

    # Exemple d'écriture (commenté)
    # request = "Create a new node with the label 'Person' and the property 'name' set to 'John Doe'."

    # Exemple de lecture
    request = "How many nodes are in the graph?"

    # Exécution de l'agent
    result = asyncio.run(run_agent(request, model))

    # Affichage du résultat
    print(result)
