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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
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
# Global Tool Results Storage
# ============================================================
# Store tool results during agent execution for source extraction
global_tool_results = {}

# ============================================================
# LangGraph State
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_results: list  # Store tool results for source extraction


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
        
        NOTE: The wrapped tool now returns the RAW dict result (not stringified)
        so that tool_execution_node can extract sources properly.
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
        # IMPORTANT: Return the raw dict, not stringified, so source extraction works
        def remote_fn(_client=client, _name=tool_name, **kwargs) -> Any:
            cleaned = {k: v for k, v in kwargs.items() if v is not None}
            try:
                result = _client.call_tool(_name, cleaned)
                # Return as-is (dict) for proper source extraction
                return result
            except Exception as exc:
                import traceback; traceback.print_exc()
                # Return error as dict structure for consistency
                return {
                    "status": "error",
                    "message": f"[{_client.name}] MCP error calling '{_name}': {exc}",
                    "sources": []
                }

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
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )
    llm_with_tools = llm.bind_tools(all_tools, tool_choice="auto")

    # ── LangGraph nodes ──────────────────────────────────────────────────────
    def agent_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

    # Custom tool execution node that captures results
    def tool_execution_node(state: State):
        """Execute tools and capture their results for source extraction."""
        global global_tool_results
        
        messages = state["messages"]
        last_message = messages[-1]
        
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {
                "messages": [],
                "tool_results": state.get("tool_results", [])
            }
        
        tool_results_messages = []
        captured_results = []
        
        # Execute each tool call
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")
            
            try:
                # Find and execute the tool
                tool_to_run = None
                for tool in all_tools:
                    if tool.name == tool_name:
                        tool_to_run = tool
                        break
                
                if tool_to_run:
                    result = tool_to_run.invoke(tool_input)
                    
                    print(f"[tool_execution] {tool_name} returned result")
                    print(f"[tool_execution] Result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"[tool_execution] Result keys: {list(result.keys())}")
                        if 'sources' in result:
                            print(f"[tool_execution] Found sources: {result['sources']}")
                    
                    # Store in global and captured list
                    tool_result_item = {
                        'tool': tool_name,
                        'result': result,
                        'full_result': result
                    }
                    captured_results.append(tool_result_item)
                    
                    # Also store globally for fallback
                    if tool_name not in global_tool_results:
                        global_tool_results[tool_name] = []
                    global_tool_results[tool_name].append(result)
                    
                    # Create ToolMessage with stringified result
                    import json
                    result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                    
                    # FIXED: Use tool_call_id instead of tool_use_id
                    tool_message = ToolMessage(
                        content=result_str,
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    tool_results_messages.append(tool_message)
                    print(f"[tool_execution] Created ToolMessage for {tool_name}")
            
            except Exception as e:
                print(f"[tool_execution] Error executing {tool_name}: {e}")
                import traceback
                traceback.print_exc()
                # Return error in dict format for consistency with source extraction
                error_result = {
                    "status": "error",
                    "message": str(e),
                    "sources": []
                }
                
                # Store error result
                tool_result_item = {
                    'tool': tool_name,
                    'result': error_result,
                    'full_result': error_result
                }
                captured_results.append(tool_result_item)
                
                if tool_name not in global_tool_results:
                    global_tool_results[tool_name] = []
                global_tool_results[tool_name].append(error_result)
                
                tool_message = ToolMessage(
                    content=str(error_result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_results_messages.append(tool_message)
        
        # Preserve existing tool_results and add new ones
        all_tool_results = state.get("tool_results", []) + captured_results
        
        return {
            "messages": tool_results_messages,
            "tool_results": all_tool_results
        }

    workflow = StateGraph(State)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_execution_node)
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

    # Add system prompt if this is a fresh conversation
    if not history:
        history.append(SystemMessage(content="""You are AgriGPT, an expert agricultural assistant.

    Provide clear, detailed, and well-formatted responses based on the available knowledge base.
    Use proper formatting with sections and bullet points for readability.
    Do NOT use markdown asterisks or special formatting characters."""))

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
    sources:      List[str] = []  # List of PDF source names from Pinecone


def extract_sources_from_result(result: Dict[str, Any]) -> List[str]:
    """
    Extract PDF source names from the agent result.
    
    Since the agent might not store ToolMessages in history,
    we extract sources by:
    1. Looking for source references in AIMessage content
    2. Parsing JSON-encoded responses
    3. Searching for PDF/document filenames in the final answer
    
    Returns: List of unique PDF filenames
    """
    sources = set()
    
    # Get the final answer text
    final_answer = None
    if "messages" in result:
        # Get the last AIMessage (final answer)
        for msg in reversed(result["messages"]):
            if hasattr(msg, '__class__') and msg.__class__.__name__ == 'AIMessage':
                if hasattr(msg, 'content'):
                    final_answer = msg.content
                    break
    
    if not final_answer:
        print("[extract_sources] No final answer found")
        return []
    
    # Parse the final answer - it might contain source info
    answer_text = ""
    if isinstance(final_answer, str):
        answer_text = final_answer
    elif isinstance(final_answer, list):
        for block in final_answer:
            if isinstance(block, dict) and 'text' in block:
                answer_text += block['text']
            elif isinstance(block, str):
                answer_text += block
    elif isinstance(final_answer, dict):
        answer_text = str(final_answer)
    
    print(f"[extract_sources] Final answer length: {len(answer_text)} chars")
    
    # Strategy 1: Look for PDF filenames mentioned in the response
    # Pattern: "filename.pdf" or filename.pdf in square brackets or parentheses
    import re
    
    # Find all PDF files mentioned
    pdf_pattern = r'[\w\s\-().]+\.pdf'
    pdf_matches = re.findall(pdf_pattern, answer_text, re.IGNORECASE)
    for pdf in pdf_matches:
        pdf_clean = pdf.strip().strip('()[]').strip()
        if pdf_clean and len(pdf_clean) > 4:  # At least "a.pdf"
            sources.add(pdf_clean)
            print(f"[extract_sources] Found PDF in answer: {pdf_clean}")
    
    # Strategy 2: Look for "Source:" or "Sources:" mentions
    source_pattern = r'(?:Source|Sources|Document)s?:?\s*([^\n]+)'
    source_matches = re.findall(source_pattern, answer_text, re.IGNORECASE)
    for match in source_matches:
        # Parse the match to extract filenames
        items = [item.strip().strip('- •*').strip() for item in match.split(',')]
        for item in items:
            item = item.strip('()[]').strip()
            if item and ('.pdf' in item.lower() or '.txt' in item.lower()):
                sources.add(item)
                print(f"[extract_sources] Found source in answer: {item}")
    
    # Strategy 3: Look for documents mentioned with extensions
    doc_pattern = r'(?:document|file|pdf)(?:\s+(?:named|called|from))?\s*["\']?([^\s"\']+\.[a-z]+)["\']?'
    doc_matches = re.findall(doc_pattern, answer_text, re.IGNORECASE)
    for doc in doc_matches:
        doc_clean = doc.strip().strip('()[]')
        if doc_clean and any(doc_clean.lower().endswith(ext) for ext in ['.pdf', '.txt', '.doc', '.docx']):
            sources.add(doc_clean)
            print(f"[extract_sources] Found document: {doc_clean}")
    
    # Strategy 4: Extract from structured responses (if any JSON is in the answer)
    try:
        # Look for JSON-like structures
        json_pattern = r'\{[^{}]*"(?:source|sources|document)s?"[^{}]*\}'
        json_matches = re.findall(json_pattern, answer_text, re.IGNORECASE)
        for json_str in json_matches:
            try:
                import json as json_module
                parsed = json_module.loads(json_str)
                if 'sources' in parsed:
                    if isinstance(parsed['sources'], list):
                        sources.update(parsed['sources'])
                elif 'source' in parsed:
                    sources.add(parsed['source'])
            except:
                pass
    except:
        pass
    
    # Clean and validate sources
    valid_sources = []
    for src in sources:
        if src and isinstance(src, str):
            src = src.strip().strip('()[]"\'')
            # Keep files with document extensions
            if any(src.lower().endswith(ext) for ext in ['.pdf', '.txt', '.doc', '.docx', '.xlsx', '.csv']):
                valid_sources.append(src)
    
    final_sources = sorted(list(set(valid_sources)))
    print(f"[extract_sources] Final extracted sources: {final_sources}")
    
    return final_sources


def extract_sources_from_tool_results(tool_results: List[Dict[str, Any]]) -> List[str]:
    """
    Extract source filenames directly from tool execution results.
    
    Tool results structure can be:
    [
        {
            'tool': 'pests_and_diseases',
            'result': {
                'status': 'success',
                'sources': [
                    {'filename': 'file1.pdf', 'chunk_id': '...', 'score': 0.8, 'text': '...'},
                    {'filename': 'file2.pdf', 'chunk_id': '...', 'score': 0.78, 'text': '...'}
                ]
            }
        }
    ]
    
    Also handles stringified JSON results and alternative structures.
    """
    sources = set()
    
    if not tool_results:
        print("[extract_sources_from_tool_results] No tool results provided")
        return []
    
    print(f"[extract_sources_from_tool_results] Processing {len(tool_results)} tool results")
    
    import json
    
    for tool_result in tool_results:
        if not isinstance(tool_result, dict):
            continue
        
        tool_name = tool_result.get("tool", "unknown")
        result_data = tool_result.get("result")
        
        if not result_data:
            print(f"[extract_sources_from_tool_results] {tool_name}: No result data")
            continue
        
        # Handle stringified JSON results (when they come from ToolMessage content)
        if isinstance(result_data, str):
            print(f"[extract_sources_from_tool_results] {tool_name}: Result is string, parsing JSON...")
            try:
                result_data = json.loads(result_data)
                print(f"[extract_sources_from_tool_results] {tool_name}: Successfully parsed JSON")
            except:
                print(f"[extract_sources_from_tool_results] {tool_name}: Could not parse as JSON, skipping")
                continue
        
        if not isinstance(result_data, dict):
            print(f"[extract_sources_from_tool_results] {tool_name}: Result is not a dict after parsing")
            continue
        
        print(f"[extract_sources_from_tool_results] {tool_name}:")
        
        # Extract from "sources" field (NEW: handles dict format with 'filename' key)
        if "sources" in result_data:
            src_list = result_data["sources"]
            if isinstance(src_list, list):
                print(f"  Found 'sources' with {len(src_list)} items")
                for src in src_list:
                    # Handle dict format: {'filename': 'file.pdf', 'chunk_id': '...', 'score': 0.8, ...}
                    if isinstance(src, dict):
                        if "filename" in src:
                            filename = src["filename"]
                            if filename and isinstance(filename, str):
                                filename = filename.strip()
                                if filename:
                                    sources.add(filename)
                                    print(f"    → {filename}")
                    # Handle plain string format: 'file.pdf'
                    elif isinstance(src, str) and src.strip():
                        sources.add(src.strip())
                        print(f"    → {src.strip()}")
        
        # Extract from "results" field with source subfield
        if "results" in result_data:
            res_list = result_data["results"]
            if isinstance(res_list, list):
                print(f"  Found 'results' with {len(res_list)} items")
                for res in res_list:
                    if isinstance(res, dict) and "source" in res:
                        src = res["source"]
                        if isinstance(src, str) and src.strip():
                            sources.add(src.strip())
                            print(f"    → {src}")
        
        # Extract from "data" or "documents" fields (alternative structures)
        if "data" in result_data and isinstance(result_data["data"], list):
            print(f"  Found 'data' field with {len(result_data['data'])} items")
            for item in result_data["data"]:
                if isinstance(item, dict) and "source" in item:
                    src = item["source"]
                    if isinstance(src, str) and src.strip():
                        sources.add(src.strip())
                        print(f"    → {src}")
    
    # Remove duplicates and sort
    final_sources = sorted(list(sources))
    print(f"[extract_sources_from_tool_results] FINAL: {final_sources}")
    return final_sources


def clean_response_text(text: str) -> str:
    """
    Clean and format the response text.
    
    Removes:
    - Asterisks (**, *, etc.) used for markdown formatting
    - 📚 Sources: section and sources (handled separately now)
    - Escaped newlines (\\n) and converts them to actual newlines
    
    Args:
        text: Raw response text from the LLM
    
    Returns:
        Cleaned, properly formatted text
    """
    if not text:
        return ""
    
    # Remove markdown formatting asterisks
    cleaned = text.replace("**", "").replace("*", "")
    
    # Convert escaped newlines to actual newlines
    cleaned = cleaned.replace("\\n", "\n")
    
    # Remove the "Sources:" section that was appended by the system prompt
    # (sources are now extracted separately)
    if "📚 Sources:" in cleaned or "Sources:" in cleaned:
        # Find the start of sources section
        if "📚 Sources:" in cleaned:
            cleaned = cleaned.split("📚 Sources:")[0]
        else:
            cleaned = cleaned.split("Sources:")[0]
    
    # Clean up extra whitespace
    cleaned = cleaned.strip()
    
    return cleaned


@app.post("/test/chat", response_model=ChatResponse)
def test_chat(request: ChatRequest):
    """
    Chat endpoint for web / mobile frontends.

    - chatId       → controls memory isolation (new UUID = blank slate)
    - phone_number → stored as metadata
    - message      → the user's input text
    
    Response includes:
    - response     → cleaned, formatted text answer
    - sources      → list of PDF filenames extracted from tool results
    """
    global global_tool_results
    
    print(f"\n[/test/chat] chatId={request.chatId} | phone={request.phone_number} | msg={request.message}")
    try:
        # Clear previous tool results for this request
        global_tool_results.clear()
        
        # Load history first
        history = load_history(request.chatId)
        print(f"[/test/chat] Loaded {len(history)} messages from history.")
        
        # IMPORTANT: Always ensure system prompt is first in history
        # to guide the agent to use tools
        system_prompt = SystemMessage(content="""You are AgriGPT, an expert agricultural assistant.

YOUR PRIMARY JOB: Call tools to retrieve information, then answer based on that.

MANDATORY RULES - FOLLOW EXACTLY:
1. Before answering ANY question, you MUST call at least ONE of these tools:
   • sme_divesh: Agricultural knowledge, AI impact, farming practices
   • pests_and_diseases: Crop diseases, pests, treatments
   • govt_schemes: Government agricultural programs and schemes
   • VignanUniversity: Academic agricultural research and information

2. WAIT for tool results. Use ONLY tool results to answer.

3. NEVER answer from your training data alone.

4. ALWAYS mention which tool(s) provided your information.

5. Format clearly without markdown asterisks.

CRITICAL: Every response must include tool calls. If you don't call a tool, you are failing your job.

Example:
User: "Tell me about AI in agriculture"
→ You: Call sme_divesh tool with query
→ Wait for results
→ Answer based on tool results only
→ Say: "According to sme_divesh tool..."

DO THIS EVERY TIME. NO EXCEPTIONS.""")
        
        # Remove any existing system messages and add fresh one
        history = [msg for msg in history if not isinstance(msg, SystemMessage)]
        history = [system_prompt] + history
        
        history.append(HumanMessage(content=request.message))
        
        # Invoke the agent
        print("[/test/chat] Invoking agent...")
        result = app_agent.invoke({
            "messages": history,
            "tool_results": []  # Initialize tool_results in state
        })
        print(f"[/test/chat] Agent returned {len(result['messages'])} messages")
        
        # Debug: Print message types and tool_calls
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            print(f"  Message {i}: {msg_type}", end="")
            
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f" → TOOL_CALLS: {[tc.get('name') for tc in msg.tool_calls]}")
            elif hasattr(msg, "content") and msg_type == "AIMessage":
                content_preview = str(msg.content)[:80]
                print(f" → Content: {content_preview}...")
            else:
                print()
        
        final_answer = extract_final_answer(result)
        
        # Save history
        save_history(request.chatId, result["messages"], phone_number=request.phone_number)
        
        # Extract sources from tool results
        print("[/test/chat] Extracting sources...")
        sources = []
        
        # Strategy 1: Try to get from result state
        if "tool_results" in result and result["tool_results"]:
            print(f"[/test/chat] Found {len(result['tool_results'])} tool results in state")
            sources = extract_sources_from_tool_results(result["tool_results"])
        
        # Strategy 2: Fallback to global storage
        if not sources and global_tool_results:
            print(f"[/test/chat] Fallback: Using global_tool_results with {len(global_tool_results)} tools")
            # Convert global format to tool_results format
            fallback_results = []
            for tool_name, results_list in global_tool_results.items():
                for result_data in results_list:
                    fallback_results.append({
                        'tool': tool_name,
                        'result': result_data
                    })
            sources = extract_sources_from_tool_results(fallback_results)
        
        # Clean and format the response text
        cleaned_response = clean_response_text(final_answer)
        
        print(f"[/test/chat] ✓ FINAL SOURCES: {sources}")
        print(f"[/test/chat] Response length: {len(cleaned_response)} chars")
        
        return ChatResponse(
            chatId=request.chatId,
            phone_number=request.phone_number,
            response=cleaned_response,
            sources=sources,
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