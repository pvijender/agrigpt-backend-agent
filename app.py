"""
FastAPI + LangGraph Agent with Multi-MCP Tool Discovery
WhatsApp Business API (Meta Cloud API) Webhook Handler

Connects to MULTIPLE MCP servers simultaneously (e.g. Alumnx + Vignan)
and merges all their tools into one agent dynamically at startup.

FALLBACK FEATURE:
  - If tools don't find relevant information in knowledge base,
    the agent makes a direct Gemini API call for the answer
  - Source information is returned in the sources array
  - Format: ["Knowledge Base: file1.pdf, file2.pdf"] or ["Not found in Knowledge Base. Used Gemini API"]

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
import json
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
MAX_MESSAGES = 20

# ============================================================
# MongoDB Setup
# ============================================================
mongo_client   = MongoClient(MONGODB_URI)
db             = mongo_client[MONGODB_DB]
chat_sessions: Collection = db[MONGODB_COLLECTION]

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
# Gemini Fallback Handler
# ============================================================
async def get_gemini_fallback_answer(user_question: str) -> str:
    """
    When knowledge base tools don't find information,
    call Gemini directly for a general answer.
    
    Returns: The generated answer text
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,  # Slightly higher for creative answers
            google_api_key=GOOGLE_API_KEY,
        )
        
        response = llm.invoke([
            SystemMessage(content="""You are AgriGPT, an expert agricultural assistant.

The user's question could not be found in the agricultural knowledge base.
Provide a helpful general answer based on your training knowledge.
Format the answer clearly without markdown asterisks."""),
            HumanMessage(content=user_question)
        ])
        
        answer_text = response.content if isinstance(response.content, str) else str(response.content)
        return answer_text
    except Exception as e:
        print(f"[Gemini Fallback] Error: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I couldn't find information about your question in the knowledge base or generate a response. Please try rephrasing your question."


# ============================================================
# MCP Client — one instance per server
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
        that build_agent() expects.
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
global_tool_results = {}

# ============================================================
# LangGraph State
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_results: list


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
        Returns RAW dict result for proper source extraction.
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

        def remote_fn(_client=client, _name=tool_name, **kwargs) -> Any:
            cleaned = {k: v for k, v in kwargs.items() if v is not None}
            try:
                result = _client.call_tool(_name, cleaned)
                return result
            except Exception as exc:
                import traceback; traceback.print_exc()
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
            print(f"[{cfg['name']}] WARNING — could not reach server: {exc}")
            continue

        for schema in remote_tools:
            raw_name     = schema["name"]
            description  = schema.get("description", "")
            input_schema = schema.get("inputSchema", {})

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
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")
            
            try:
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
                    
                    tool_result_item = {
                        'tool': tool_name,
                        'result': result,
                        'full_result': result
                    }
                    captured_results.append(tool_result_item)
                    
                    if tool_name not in global_tool_results:
                        global_tool_results[tool_name] = []
                    global_tool_results[tool_name].append(result)
                    
                    result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                    
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
                error_result = {
                    "status": "error",
                    "message": str(e),
                    "sources": []
                }
                
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


async def run_agent(chat_id: str, user_message: str, phone_number: str | None = None) -> Dict[str, Any]:
    """
    Single entry point for agent execution across all channels.

    Flow:
      1. Load history for chat_id from MongoDB.
      2. Append the new human message.
      3. Invoke the LLM with the full message history as context.
      4. Check if knowledge base found relevant results.
      5. Save updated history back to MongoDB.
      6. Return answer + sources info.
    """
    print(f"[run_agent] chat_id={chat_id} | phone={phone_number} | msg={user_message[:60]}")

    history = load_history(chat_id)
    print(f"[run_agent] Loaded {len(history)} messages from history.")

    # Add system prompt if this is a fresh conversation
    if not history:
        history.append(SystemMessage(content="""You are AgriGPT, an expert agricultural assistant.

YOUR PRIMARY JOB: Call tools to retrieve information from the knowledge base.

MANDATORY RULES (FOLLOW EXACTLY):
1. EVERY question, you MUST call at least ONE tool first:
   • sme_divesh: Agricultural knowledge, AI impact, farming practices
   • pests_and_diseases: Crop diseases, pests, treatments
   • govt_schemes: Government agricultural programs and schemes
   • VignanUniversity: Academic agricultural research

2. WAIT for tool results to come back.

3. If tool results are EMPTY or contain "not found" or "no results":
   - DO NOT answer from your training knowledge
   - Tell the user: "Not found in knowledge base"
   - Include this phrase EXACTLY in your response

4. If tools return actual information:
   - Answer ONLY based on that tool information

5. Format answers clearly without markdown asterisks.

CRITICAL: Never answer from your training data alone. Always call tools first.
If tools have no results, say "Not found in knowledge base" in your response."""))

    history.append(HumanMessage(content=user_message))

    result = app_agent.invoke({
        "messages": history,
        "tool_results": []
    })
    
    final_answer = extract_final_answer(result)

    save_history(chat_id, result["messages"], phone_number=phone_number)
    print(f"[run_agent] Saved history. Answer: {final_answer[:80]}")

    return {
        "answer": final_answer,
        "tool_results": result.get("tool_results", [])
    }


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
# Background Task — WhatsApp channel
# ============================================================
async def process_and_reply(phone_number: str, user_message: str):
    """
    For WhatsApp: chat_id == phone_number (one persistent session per number).
    """
    try:
        result = await run_agent(phone_number, user_message, phone_number)
        final_answer = result["answer"]
        print(f"[WhatsApp] Reply for {phone_number}: {final_answer[:100]}")
        print("[WhatsApp] Send skipped (LOCAL MODE).")
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[WhatsApp] Error for {phone_number}: {exc}")


# ============================================================
# Chat Endpoint — Web / Mobile Frontend
# ============================================================
class ChatRequest(BaseModel):
    chatId:       str
    phone_number: str
    message:      str


class ChatResponse(BaseModel):
    chatId:       str
    phone_number: str
    response:     str
    sources:      List[str] = []


def extract_sources_from_tool_results(tool_results: List[Dict[str, Any]]) -> List[str]:
    """
    Extract source filenames from tool execution results.
    
    Returns either:
    - List of PDF filenames if found in KB
    - ["Not found in Knowledge Base. Used Gemini API"] if KB had no results
    """
    sources = set()
    
    if not tool_results:
        print("[extract_sources] No tool results provided")
        return []
    
    print(f"[extract_sources] Processing {len(tool_results)} tool results")
    
    for tool_result in tool_results:
        if not isinstance(tool_result, dict):
            continue
        
        tool_name = tool_result.get("tool", "unknown")
        result_data = tool_result.get("result")
        
        if not result_data:
            print(f"[extract_sources] {tool_name}: No result data")
            continue
        
        # Handle stringified JSON results
        if isinstance(result_data, str):
            try:
                result_data = json.loads(result_data)
            except:
                print(f"[extract_sources] {tool_name}: Could not parse as JSON")
                continue
        
        if not isinstance(result_data, dict):
            continue
        
        # Extract from "sources" field
        if "sources" in result_data:
            src_list = result_data["sources"]
            if isinstance(src_list, list):
                for src in src_list:
                    if isinstance(src, dict) and "filename" in src:
                        filename = src["filename"]
                        if filename and isinstance(filename, str):
                            sources.add(filename.strip())
                    elif isinstance(src, str) and src.strip():
                        sources.add(src.strip())
        
        # Extract from "results" field
        if "results" in result_data:
            res_list = result_data["results"]
            if isinstance(res_list, list):
                for res in res_list:
                    if isinstance(res, dict) and "source" in res:
                        src = res["source"]
                        if isinstance(src, str) and src.strip():
                            sources.add(src.strip())
    
    final_sources = sorted(list(sources))
    print(f"[extract_sources] Found {len(final_sources)} sources")
    
    return final_sources


def clean_response_text(text: str) -> str:
    """
    Clean and format the response text.
    """
    if not text:
        return ""
    
    # Remove markdown formatting asterisks
    cleaned = text.replace("**", "").replace("*", "")
    
    # Convert escaped newlines to actual newlines
    cleaned = cleaned.replace("\\n", "\n")
    
    # Remove source sections if any
    if "📚 Sources:" in cleaned or "Sources:" in cleaned:
        if "📚 Sources:" in cleaned:
            cleaned = cleaned.split("📚 Sources:")[0]
        else:
            cleaned = cleaned.split("Sources:")[0]
    
    cleaned = cleaned.strip()
    
    return cleaned


@app.post("/test/chat", response_model=ChatResponse)
async def test_chat(request: ChatRequest):
    """
    Chat endpoint for web / mobile frontends.
    
    Returns response with sources array containing either:
    - List of PDF files (if found in KB)
    - ["Not found in Knowledge Base. Used Gemini API"] if KB didn't find answer
    """
    global global_tool_results
    
    print(f"\n[/test/chat] chatId={request.chatId} | phone={request.phone_number} | msg={request.message}")
    try:
        global_tool_results.clear()
        
        # Run agent
        agent_result = await run_agent(
            request.chatId,
            request.message,
            request.phone_number
        )
        
        final_answer = agent_result["answer"]
        tool_results = agent_result.get("tool_results", [])
        
        # Extract sources from tool results
        print("[/test/chat] Extracting sources...")
        sources = []
        
        # Check if KB found any results
        kb_found = False
        if tool_results:
            # Convert to proper format
            fallback_results = []
            for tool_result in tool_results:
                fallback_results.append(tool_result)
            
            sources = extract_sources_from_tool_results(fallback_results)
            if sources:
                kb_found = True
        
        # If KB didn't find anything, check if answer contains "not found"
        if not kb_found and ("not found" in final_answer.lower() or 
                             "not available" in final_answer.lower() or 
                             "no information" in final_answer.lower()):
            # Call Gemini fallback
            print("[/test/chat] KB didn't find answer, calling Gemini fallback...")
            gemini_answer = await get_gemini_fallback_answer(request.message)
            # Prepend the disclaimer message
            final_answer = f"Sorry not found in knowledge base but I can use Gemini API to answer your question\n\n{gemini_answer}"
            sources = ["Gemini"]
        elif not sources and ("not found" in final_answer.lower() or 
                              "not available" in final_answer.lower() or
                              "no results" in final_answer.lower() or
                              "no data" in final_answer.lower()):
            # KB tools were called but returned nothing
            print("[/test/chat] KB tools returned no results, calling Gemini...")
            gemini_answer = await get_gemini_fallback_answer(request.message)
            # Prepend the disclaimer message
            final_answer = f"Sorry not found in knowledge base but I can use Gemini API to answer your question\n\n{gemini_answer}"
            sources = ["Gemini"]
        elif not sources and not kb_found:
            # Tools were called but returned no meaningful results
            # This catches cases where agent answers without calling tools or tools return empty
            print("[/test/chat] KB didn't return sources, using Gemini fallback...")
            gemini_answer = await get_gemini_fallback_answer(request.message)
            # Prepend the disclaimer message
            final_answer = f"Sorry not found in knowledge base but I can use Gemini API to answer your question\n\n{gemini_answer}"
            sources = ["Gemini"]
        
        # Clean response
        cleaned_response = clean_response_text(final_answer)
        
        print(f"[/test/chat] Final sources: {sources}")
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