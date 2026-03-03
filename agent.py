"""
FastAPI + LangGraph Agent with Multi-MCP Tool Discovery
WhatsApp Business API (Meta Cloud API) Webhook Handler

Connects to MULTIPLE MCP servers simultaneously (e.g. Alumnx + Vignan)
and merges all their tools into one agent dynamically at startup.

New Chat flow:
  - Frontend generates a new UUID on "New Chat" click and sends it as chat_id.
  - Backend finds no history for that chat_id → agent starts fresh.
  - MongoDB creates the document automatically on first save.
  - Same chat_id on subsequent messages → history is loaded and agent remembers.

Auto Deploy enabled using deploy.yml file
"""

import os
import httpx
import asyncio
from datetime import datetime, timezone
from typing import Annotated, TypedDict, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection

# ============================================================
# Environment
# ============================================================
load_dotenv()

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"]    = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"]    = "agrigpt-backend-agent"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MCP_TIMEOUT    = float(os.getenv("MCP_TIMEOUT", "30"))

# ── Multi-MCP Configuration ──────────────────────────────────────────────────
# Each MCP server is configured via its own pair of env vars:
#   <NAME>_MCP_URL      → base URL of that server
#   <NAME>_MCP_API_KEY  → optional Bearer token (leave blank if not needed)
#
# The agent contacts ALL servers at startup, discovers their tools, and
# merges everything into a single LangGraph agent automatically.
# To add a third server later, just add its env vars and a new entry below.
# ─────────────────────────────────────────────────────────────────────────────
MCP_SERVERS: List[Dict[str, str]] = [
    {
        "name":    "Alumnx",
        "url":     os.getenv("ALUMNX_MCP_URL", "http://localhost:9000"),
        "api_key": os.getenv("ALUMNX_MCP_API_KEY", ""),
    },
    {
        "name":    "Vignan",
        "url":     os.getenv("VIGNAN_MCP_URL", "http://localhost:8000"),
        "api_key": os.getenv("VIGNAN_MCP_API_KEY", ""),
    },
]

MONGODB_URI        = os.getenv("MONGODB_URI")
MONGODB_DB         = os.getenv("MONGODB_DB", "agrigpt")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chats")

# Max messages stored per chat_id (human + AI combined = 10 full turns).
# The LLM receives ALL stored messages as context on every invocation.
MAX_MESSAGES = 20

# WHATSAPP: Uncomment when Meta credentials are ready
# WHATSAPP_VERIFY_TOKEN    = os.getenv("WHATSAPP_VERIFY_TOKEN")
# WHATSAPP_ACCESS_TOKEN    = os.getenv("WHATSAPP_ACCESS_TOKEN")
# WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

# ============================================================
# MongoDB Setup
# ============================================================
mongo_client   = MongoClient(MONGODB_URI)
db             = mongo_client[MONGODB_DB]
chat_sessions: Collection = db[MONGODB_COLLECTION]

# chat_id      → unique  (one document per conversation session)
# phone_number → non-unique (one user can have many chat sessions)
# updated_at   → for future TTL / cleanup
chat_sessions.create_index([("chat_id",      ASCENDING)], unique=True)
chat_sessions.create_index([("phone_number", ASCENDING)])
chat_sessions.create_index([("updated_at",   ASCENDING)])

print(f"Connected to MongoDB: {MONGODB_DB}.{MONGODB_COLLECTION}")

# ============================================================
# MongoDB Memory Helpers
# ============================================================

def load_history(chat_id: str) -> list:
    """
    Load stored messages for a chat session and reconstruct LangChain
    message objects.

    Returns all stored messages (up to MAX_MESSAGES). The agent feeds
    ALL of them to the LLM so it can answer new questions with full
    awareness of the entire conversation history for that chat_id.

    If chat_id is new (no document exists) → returns empty list
    → agent starts a fresh conversation automatically.
    """
    doc = chat_sessions.find_one({"chat_id": chat_id})
    if not doc or "messages" not in doc:
        return []

    reconstructed = []
    for m in doc["messages"]:
        role    = m.get("role")
        content = m.get("content", "")
        if role == "human":
            reconstructed.append(HumanMessage(content=content))
        elif role == "ai":
            reconstructed.append(AIMessage(content=content))
        elif role == "system":
            reconstructed.append(SystemMessage(content=content))
    return reconstructed


def save_history(chat_id: str, messages: list, phone_number: str | None = None):
    """
    Persist updated conversation history to MongoDB under chat_id.

    Steps:
      1. Strip ToolMessages and tool-call-only AIMessages (not useful as LLM context).
      2. Apply pair-aware sliding window: keep the last MAX_MESSAGES messages,
         always ending on a complete human+AI pair.
      3. Upsert the document — creates it on first save (new chat),
         updates it on subsequent saves (continuing chat).
    """
    # Step 1 — filter to storable human/ai messages only
    storable = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            storable.append({"role": "human", "content": content})

        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                storable.append({"role": "ai", "content": content})
            elif isinstance(content, list):
                text_parts = [
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ]
                joined = " ".join(t for t in text_parts if t.strip())
                if joined.strip():
                    storable.append({"role": "ai", "content": joined})
        # ToolMessage and other internal types are intentionally skipped

    # Step 2 — pair-aware sliding window
    if len(storable) <= MAX_MESSAGES:
        window = storable
    else:
        pairs_to_collect = MAX_MESSAGES // 2
        pairs_collected  = 0
        cutoff_index     = 0
        i = len(storable) - 1

        while i >= 0 and pairs_collected < pairs_to_collect:
            if storable[i]["role"] == "ai" and i > 0 and storable[i - 1]["role"] == "human":
                pairs_collected += 1
                cutoff_index = i - 1
                i -= 2
            else:
                i -= 1

        window = storable[cutoff_index:] if pairs_collected > 0 else storable[-MAX_MESSAGES:]

    # Step 3 — upsert
    now = datetime.now(timezone.utc)
    update_fields: dict = {
        "messages":   window,
        "updated_at": now,
    }
    if phone_number:
        update_fields["phone_number"] = phone_number

    chat_sessions.update_one(
        {"chat_id": chat_id},
        {
            "$set":         update_fields,
            "$setOnInsert": {"created_at": now},
        },
        upsert=True
    )


# ============================================================
# MCP Client — one instance per server
#
# Your MCP servers expose this custom REST API:
#   GET  /list-tools  → {
#                         "tools": [{
#                           "name": "...",
#                           "description": "...",
#                           "parameters": {
#                             "param_name": {
#                               "type": "string",
#                               "required": true/false,   ← inline bool, NOT a top-level array
#                               "default": "...",
#                               "description": "..."
#                             }
#                           }
#                         }]
#                       }
#   POST /callTool    → { "name": "...", "arguments": {...} }
#                     ← { "result": ... }
# ============================================================
class MCPClient:
    """REST client matching your MCP servers' custom endpoint format."""

    def __init__(self, name: str, base_url: str, api_key: str | None = None):
        self.name     = name
        self.base_url = base_url.rstrip("/")
        self.headers  = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(timeout=MCP_TIMEOUT)

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        GET /list-tools and normalize the response into the internal format
        that build_agent() expects:
          { name, description, inputSchema: { properties: {...}, required: [...] } }

        The server returns a flat "parameters" dict where each param carries
        an inline "required" boolean. We convert that to a standard JSON Schema
        shape so the rest of the agent code doesn't need to know about it.
        """
        print(f"[{self.name}] Fetching tools → {self.base_url}/list-tools")
        response = self.client.get(
            f"{self.base_url}/list-tools",
            headers=self.headers,
        )
        response.raise_for_status()
        raw_tools: List[Dict] = response.json().get("tools", [])

        normalized = []
        for tool in raw_tools:
            params     = tool.get("parameters", {})
            properties = {}
            required   = []

            for prop_name, prop_details in params.items():
                properties[prop_name] = {
                    "type":        prop_details.get("type", "string"),
                    "description": prop_details.get("description", ""),
                    "default":     prop_details.get("default", None),
                }
                # Server uses inline "required": true/false on each param
                if prop_details.get("required", False):
                    required.append(prop_name)

            normalized.append({
                "name":        tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": {
                    "properties": properties,
                    "required":   required,
                },
            })

        print(f"[{self.name}] Found {len(normalized)} tool(s): {[t['name'] for t in normalized]}")
        return normalized

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        print(f"[{self.name}] Calling '{name}' | args: {arguments}")
        response = self.client.post(
            f"{self.base_url}/callTool",
            headers=self.headers,
            json={"name": name, "arguments": arguments},
        )
        response.raise_for_status()
        result = response.json().get("result")
        print(f"[{self.name}] Result: {str(result)[:300]}")
        return result


# ============================================================
# LangGraph State
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# Agent Builder — discovers & merges tools from ALL MCP servers
# ============================================================
def build_agent():
    TYPE_MAP = {
        "string":  str,
        "integer": int,
        "number":  float,
        "boolean": bool,
        "array":   list,
        "object":  dict,
    }

    def wrap_tool(
        client: MCPClient,
        tool_name: str,
        description: str,
        input_schema: Dict[str, Any],
    ) -> StructuredTool:
        """
        Wrap a single remote MCP tool as a LangChain StructuredTool.
        `client` and `tool_name` are captured explicitly via default
        arguments so every tool dispatches to the correct server even
        when created inside a loop.
        """
        properties      = input_schema.get("properties", {})
        required_fields = set(input_schema.get("required", []))
        field_defs      = {}

        for prop_name, prop_details in properties.items():
            py_type   = TYPE_MAP.get(prop_details.get("type", "string"), str)
            prop_desc = prop_details.get("description", "")
            if prop_name in required_fields:
                field_defs[prop_name] = (py_type, Field(..., description=prop_desc))
            else:
                field_defs[prop_name] = (
                    py_type,
                    Field(default=prop_details.get("default", None), description=prop_desc),
                )

        ArgsSchema = create_model(f"{tool_name}_args", **field_defs)

        # Default-argument capture prevents late-binding bugs in loops
        def remote_fn(_client=client, _name=tool_name, **kwargs) -> str:
            cleaned = {k: v for k, v in kwargs.items() if v is not None}
            try:
                return str(_client.call_tool(_name, cleaned))
            except Exception as exc:
                import traceback; traceback.print_exc()
                return f"[{_client.name}] MCP error calling '{_name}': {exc}"

        return StructuredTool.from_function(
            func=remote_fn,
            name=tool_name,
            description=f"[{client.name}] {description}",
            args_schema=ArgsSchema,
        )

    # ── Discover tools from every configured MCP server ──────────────────────
    all_tools:  List[StructuredTool] = []
    seen_names: set                  = set()

    for cfg in MCP_SERVERS:
        client = MCPClient(
            name=cfg["name"],
            base_url=cfg["url"],
            api_key=cfg.get("api_key") or None,
        )
        try:
            remote_tools = client.list_tools()
        except Exception as exc:
            # One unreachable server must NOT crash the whole agent at startup
            print(f"[{cfg['name']}] WARNING — could not reach server: {exc}")
            continue

        for schema in remote_tools:
            raw_name     = schema["name"]
            description  = schema.get("description", "")
            input_schema = schema.get("inputSchema", {})

            # Prefix with server name if two servers share the same tool name
            unique_name = raw_name
            if raw_name in seen_names:
                unique_name = f"{cfg['name'].lower()}_{raw_name}"
                print(
                    f"[{cfg['name']}] Duplicate tool name '{raw_name}' "
                    f"→ renamed to '{unique_name}'"
                )
            seen_names.add(unique_name)

            all_tools.append(wrap_tool(client, unique_name, description, input_schema))

    if not all_tools:
        raise RuntimeError(
            "No tools discovered from any MCP server. "
            "Check that ALUMNX_MCP_URL and VIGNAN_MCP_URL are reachable."
        )

    print(f"\n✅ Total tools loaded: {len(all_tools)}")
    print(f"   Tool names: {[t.name for t in all_tools]}\n")

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )
    llm_with_tools = llm.bind_tools(all_tools)

    # ── LangGraph nodes ──────────────────────────────────────────────────────
    def agent_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

    workflow = StateGraph(State)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(all_tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


# ============================================================
# Startup — build the agent once at process start
# ============================================================
print("\nBUILDING AGENT AT STARTUP...")
app_agent = build_agent()
print("AGENT BUILD COMPLETE\n")


# ============================================================
# Core Agent Invocation — shared by ALL channels
# ============================================================
def extract_final_answer(result: dict) -> str:
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, str) and msg.content.strip():
                return msg.content
            elif isinstance(msg.content, list) and msg.content:
                block = msg.content[0]
                if isinstance(block, dict) and block.get("text", "").strip():
                    return block["text"]
                elif str(block).strip():
                    return str(block)
    return "No response generated."


def run_agent(chat_id: str, user_message: str, phone_number: str | None = None) -> str:
    """
    Single entry point for agent execution across all channels (web, WhatsApp).

    Flow:
      1. Load history for chat_id from MongoDB.
         - New chat_id → empty history → fresh conversation.
         - Existing chat_id → full history → agent remembers previous context.
      2. Append the new human message.
      3. Invoke the LLM with the full message history as context.
      4. Save updated history back to MongoDB (trimmed to MAX_MESSAGES).
      5. Return the final text answer.
    """
    print(f"[run_agent] chat_id={chat_id} | phone={phone_number} | msg={user_message[:60]}")

    history = load_history(chat_id)
    print(f"[run_agent] Loaded {len(history)} messages from history.")

    history.append(HumanMessage(content=user_message))

    result       = app_agent.invoke({"messages": history})
    final_answer = extract_final_answer(result)

    save_history(chat_id, result["messages"], phone_number=phone_number)
    print(f"[run_agent] Saved history. Answer: {final_answer[:80]}")

    return final_answer


# ============================================================
# WhatsApp Sender (uncomment when Meta credentials are ready)
# ============================================================
# async def send_whatsapp_message(to_phone: str, message: str):
#     url     = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
#     headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}", "Content-Type": "application/json"}
#     payload = {"messaging_product": "whatsapp", "to": to_phone, "type": "text", "text": {"body": message}}
#     async with httpx.AsyncClient(timeout=10.0) as client:
#         resp = await client.post(url, headers=headers, json=payload)
#         if resp.status_code != 200:
#             print(f"Failed to send WhatsApp message: {resp.text}")


# ============================================================
# Background Task — WhatsApp channel
# ============================================================
async def process_and_reply(phone_number: str, user_message: str):
    """
    For WhatsApp: chat_id == phone_number (one persistent session per number).
    Runs after 200 OK is returned to the WhatsApp webhook.
    """
    try:
        loop         = asyncio.get_event_loop()
        final_answer = await loop.run_in_executor(
            None, run_agent, phone_number, user_message, phone_number
        )
        print(f"[WhatsApp] Reply for {phone_number}: {final_answer[:100]}")
        # await send_whatsapp_message(phone_number, final_answer)
        print("[WhatsApp] Send skipped (LOCAL MODE).")
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[WhatsApp] Error for {phone_number}: {exc}")


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="AgriGPT Agent")


# ============================================================
# WhatsApp Webhook Verification (GET)
# ============================================================
@app.get("/webhook")
async def verify_webhook(
    hub_mode:         str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge:    str = Query(None, alias="hub.challenge"),
):
    # WHATSAPP: replace hardcoded token with WHATSAPP_VERIFY_TOKEN env var when going live
    LOCAL_VERIFY_TOKEN = "test_verify_token_123"
    if hub_mode == "subscribe" and hub_verify_token == LOCAL_VERIFY_TOKEN:
        print("Webhook verified successfully.")
        return PlainTextResponse(content=hub_challenge, status_code=200)
    raise HTTPException(status_code=403, detail="Webhook verification failed.")


# ============================================================
# WhatsApp Webhook Handler (POST)
# ============================================================
@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives WhatsApp events. Returns 200 immediately, processes in background."""
    payload = await request.json()
    print(f"[Webhook] Incoming payload: {payload}")
    try:
        entry    = payload.get("entry", [{}])[0]
        changes  = entry.get("changes", [{}])[0]
        value    = changes.get("value", {})
        messages = value.get("messages", [])

        if not messages:
            return {"status": "ok"}

        message  = messages[0]
        msg_type = message.get("type")
        if msg_type != "text":
            print(f"[Webhook] Ignoring non-text type: {msg_type}")
            return {"status": "ok"}

        phone_number = message.get("from")
        user_message = message["text"].get("body", "").strip()
        if not phone_number or not user_message:
            return {"status": "ok"}

        print(f"[Webhook] Message from {phone_number}: {user_message}")
        background_tasks.add_task(process_and_reply, phone_number, user_message)

    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[Webhook] Parse error: {exc}")

    return {"status": "ok"}


# ============================================================
# Chat Endpoint — Web / Mobile Frontend
#
# Frontend contract:
#   • On "New Chat" click → generate a fresh UUID and store it:
#       const chatId = crypto.randomUUID()          // browser
#       import { v4 as uuidv4 } from 'uuid'         // Node / React Native
#
#   • Send chat_id + phone_number + message on every turn of that session.
#   • On next "New Chat" click → generate a new UUID → fresh conversation.
#
# Backend behaviour:
#   • New chat_id → no history found → agent starts completely fresh.
#   • Same chat_id → history loaded → agent answers with full context.
#   • MongoDB document created automatically on first message of a new chat.
# ============================================================
class ChatRequest(BaseModel):
    chatId:       str   # UUID generated by frontend — new UUID = new conversation
    phone_number: str   # user's phone number — stored as metadata
    message:      str   # user's message text


class ChatResponse(BaseModel):
    chatId:       str
    phone_number: str
    response:     str


@app.post("/test/chat", response_model=ChatResponse)
def test_chat(request: ChatRequest):
    """
    Chat endpoint for web / mobile frontends.

    - chatId       → controls memory isolation (new UUID = blank slate)
    - phone_number → stored as metadata
    - message      → the user's input text
    """
    print(f"\n[/test/chat] chatId={request.chatId} | phone={request.phone_number} | msg={request.message}")
    try:
        final_answer = run_agent(
            chat_id=request.chatId,
            user_message=request.message,
            phone_number=request.phone_number,
        )
        return ChatResponse(
            chatId=request.chatId,
            phone_number=request.phone_number,
            response=final_answer,
        )
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)