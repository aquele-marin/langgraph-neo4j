import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
# from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações do Grafo e LLM
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Inicializa a conexão com o banco de dados Neo4j
# LangChain usa esta conexão para entender o schema do grafo (nós, relações, propriedades)
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Inicializa o Modelo de Linguagem (LLM) que irá gerar as consultas Cypher
# Usamos um modelo com alta capacidade de seguir instruções como o gpt-4
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=2)

# Cria a "Chain" de Q&A. Esta é a peça central da LangChain aqui.
# Ela recebe uma pergunta, inspeciona o schema do grafo, pede ao LLM para criar
# uma consulta Cypher, executa a consulta e retorna a resposta.
chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    verbose=True, # verbose=True é ÓTIMO para debug, pois mostra a consulta gerada
    allow_dangerous_requests=True
)

def ask_question(question):
    """Função para fazer uma pergunta ao nosso sistema de Q&A."""
    print(f"\n> Pergunta: {question}")
    result = chain.invoke({"query": question})
    print("\n< Resposta:")
    print(result['result'])


# --- Vamos testar! ---
ask_question("Who directed the movie The Matrix?")
ask_question("What movies did Tom Hanks act in?")
ask_question("How many movies are in the database?")
