# agrigpt-backend-agent
This repository contains the code for the Agent which powers the AgriGPT application by understanding the intent of the user

## Table of Contents

- [System Design](#system-design)
- [Backend Service](#backend-service)

### System Design
The system design documented at :
https://app.eraser.io/workspace/EosqvWmxFExAt23okqUH


### Backend Service

Repository:
https://github.com/alumnx-ai-labs/agrigpt-backend-agent.git

Deployed Url: 
https://newapi.alumnx.com/agrigpt/agent/docs


### Refer below google doc for more details
>https://docs.google.com/document/d/11KcYh65rblBxE4t45g55jalRWydqzOPqzIhnQmYYH6U/edit?tab=t.0

=======
# AgriGPT Backend Agent

A FastAPI-based conversational AI agent for agricultural assistance, powered by Google Gemini and LangGraph. Supports multi-turn chat via a web/mobile frontend and is designed for future WhatsApp Business API integration.

---

## Features

- **Multi-turn memory** — conversation history stored per `chat_id` in MongoDB; up to 20 messages (10 full turns) retained with a pair-aware sliding window.
- **Dynamic MCP tool discovery** — tools are fetched from a remote MCP server at startup; the agent calls them automatically as needed.
- **Multi-channel ready** — shared `run_agent()` core used by both the web frontend endpoint and the WhatsApp webhook handler.
- **Session isolation** — each `chat_id` UUID is an independent conversation; one phone number can have many parallel sessions.
- **LangSmith tracing** — optional; enabled automatically when `LANGSMITH_API_KEY` is present.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| LLM | Google Gemini 2.5 Flash Lite (`langchain-google-genai`) |
| Agent Orchestration | LangGraph |
| Tool Protocol | Remote MCP (HTTP JSON) |
| Database | MongoDB (PyMongo) |
| HTTP Client | httpx |
| Tracing (optional) | LangSmith |

---

## Project Structure

```
.
├── app.py          # Main application — all agent, API, and DB logic
├── deploy.yml      # Auto-deploy configuration
├── .env            # Environment variables (not committed)
└── README.md
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your_google_api_key
MONGODB_URI=mongodb_url
LANGSMITH_API_KEY=your_langsmith_api_key


# Optional — defaults shown
MCP_BASE_URL=your_mcp_url
MCP_API_KEY=your_mcp_api_key
MONGODB_DB=database_name
MONGODB_COLLECTION=database_collection




```

---

## Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd agrigpt-backend

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install fastapi uvicorn httpx python-dotenv pymongo \
            langgraph langchain-core langchain-google-genai \
            pydantic

# 4. Configure environment
cp .env.example .env
# Edit .env with your credentials

# 5. Run the server
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8020 --reload
```

Server starts on: `http://localhost:8020`  
Interactive API docs: `http://localhost:8020/docs`

---

## API Reference

### `POST /test/chat`

Main chat endpoint for web and mobile frontends.

**Request Body:**
```json
{
  "chat_id":      "550e8400-e29b-41d4-a716-446655440000",
  "phone_number": "911234567890",
  "message":      "Which crops are best for black soil?"
}

As minor change has been made here
