# IPL RAG Chat — Spring Boot + LangChain + Chroma

A minimal end-to-end RAG chat app:

- **Spring Boot (Java 17)** serves the chat UI (`/`) and exposes ONE API (`POST /api/chat`).
- **Python FastAPI microservice** runs the LangChain + HuggingFace (`all-MiniLM-L6-v2`) + Chroma RAG pipeline over 10 IPL team documents.
- Spring Boot forwards prompts to the Python service and returns the top-k retrieved documents.

## Project Structure

```
rag-chat/
├── pom.xml
├── pyrag/
│   ├── rag_service.py         # Python RAG microservice (FastAPI)
│   └── requirements.txt
└── src/main/
    ├── java/com/example/ragchat/
    │   ├── RagChatApplication.java
    │   └── ChatController.java
    └── resources/
        ├── application.properties
        └── static/
            └── index.html     # Chat UI (plain HTML/JS)
```

## Prerequisites

- Java 17+
- Maven 3.6+
- Python 3.10+
- ~500 MB free disk (HuggingFace model + Chroma DB)

## How to Run

Open **two terminals**.

### Terminal 1 — Python RAG microservice (port 5005)

```bash
cd pyrag
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn rag_service:app --host 0.0.0.0 --port 5005
```

Wait for `Application startup complete.` — the first run downloads the HuggingFace embedding model and seeds the Chroma vector store.

### Terminal 2 — Spring Boot (port 8080)

```bash
mvn spring-boot:run
```

Or build & run the jar:

```bash
mvn clean package -DskipTests
java -jar target/rag-chat.jar
```

### Open the app

Visit **http://localhost:8080** and start chatting.

Example prompts:
- "Which team has strong all-rounders with spin bowling?"
- "Who is the captain of Chennai Super Kings?"
- "Tell me about Mumbai Indians."

## API

`POST /api/chat`

Request:
```json
{ "prompt": "Which team has strong all-rounders with spin bowling?", "k": 3 }
```

Response:
```json
{
  "query": "Which team has strong all-rounders with spin bowling?",
  "results": [
    { "team": "Delhi Capitals", "content": "..." },
    { "team": "Rajasthan Royals", "content": "..." },
    { "team": "Gujarat Titans", "content": "..." }
  ]
}
```

## Architecture

```
Browser (index.html)
   │  POST /api/chat  { prompt, k:3 }
   ▼
Spring Boot ChatController  (port 8080)
   │  WebClient → POST /rag/query { query, k:3 }
   ▼
Python FastAPI (port 5005)
   │  vector_store.similarity_search(query, k=3)
   ▼
Chroma DB + HuggingFace all-MiniLM-L6-v2 embeddings
```

## Config

Override the RAG service URL if you run Python on a different host/port:

```bash
export RAG_SERVICE_URL=http://127.0.0.1:5005
mvn spring-boot:run
```
