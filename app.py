"""
FastAPI + LangGraph Agent with Multi-MCP Tool Discovery
WhatsApp Business API (Meta Cloud API) Webhook Handler

FIXED VERSION: Uses Mohan's proven approach to capture and extract sources correctly.

Connects to MULTIPLE MCP servers simultaneously (e.g. Alumnx + Vignan)
and merges all their tools into one agent dynamically at startup.

CRITICAL: Agent ALWAYS calls tools first before answering.
System prompt forces tool usage with mandatory rules.

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
        "url":     os.getenv("ALUMNX_MCP_URL", "").strip(),
        "api_key": os.getenv("ALUMNX_MCP_API_KEY", "").strip(),
    },
    {
        "name":    "Vignan",
        "url":     os.getenv("VIGNAN_MCP_URL", "").strip(),
        "api_key": os.getenv("VIGNAN_MCP_API_KEY", "").strip(),
    },
]

MONGODB_URI        = os.getenv("MONGODB_URI")
MONGODB_DB         = os.getenv("MONGODB_DB", "agrigpt")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chats")

MAX_MESSAGES = 20

# ============================================================
# GLOBAL STORAGE FOR RAW TOOL RESULTS
# This is the KEY FIX from Mohan's approach
# Stores raw dict results BEFORE any stringification
# ============================================================
global_tool_results: List[Dict[str, Any]] = []

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
    """Load stored messages for a chat session."""
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
    """Persist updated conversation history to MongoDB under chat_id."""
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

    # Pair-aware sliding window
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

    # Upsert
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
# MCP Client
# ============================================================
class MCPClient:
    """REST client matching MCP servers' custom endpoint format."""

    def __init__(self, name: str, base_url: str, api_key: str | None = None):
        self.name     = name
        self.base_url = base_url.rstrip("/")
        self.headers  = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(timeout=MCP_TIMEOUT)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Fetch tools from MCP server."""
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
# LangGraph State
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# Agent Builder
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
        """Wrap a single remote MCP tool as a LangChain StructuredTool."""
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

    # Discover tools from every configured MCP server
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

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )
    llm_with_tools = llm.bind_tools(all_tools, tool_choice="auto")

    # LangGraph nodes
    def agent_node(state: State):
        return {
            "messages": [llm_with_tools.invoke(state["messages"])],
        }

    def should_continue(state: State):
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

    # ========== KEY FIX: Tool execution node that captures RAW results ==========
    def tool_execution_node(state: State):
        """Execute tools and capture RAW results to global_tool_results."""
        global global_tool_results
        
        messages = state["messages"]
        last_message = messages[-1]
        
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": []}
        
        tool_results_messages = []
        
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
                    # ✅ CAPTURE RAW RESULT BEFORE STRINGIFICATION
                    result = tool_to_run.invoke(tool_input)
                    
                    print(f"[tool_execution] {tool_name} returned result")
                    print(f"[tool_execution] Result type: {type(result)}")
                    
                    # Debug: Log the actual structure for source extraction
                    if isinstance(result, list) and len(result) > 0:
                        print(f"[tool_execution] Result is list, first item keys: {result[0].keys() if isinstance(result[0], dict) else 'N/A'}")
                        print(f"[tool_execution] First item sample: {str(result[0])[:200]}")
                    elif isinstance(result, dict):
                        print(f"[tool_execution] Result is dict, keys: {result.keys()}")
                    
                    # ✅ STORE RAW DICT IN GLOBAL (THIS IS THE KEY FIX)
                    tool_result_item = {
                        'tool': tool_name,
                        'result': result,        # ← RAW DICT
                        'full_result': result    # ← PRESERVED FOR SOURCE EXTRACTION
                    }
                    global_tool_results.append(tool_result_item)
                    print(f"[tool_execution] Stored raw result for {tool_name}")
                    
                    # Create ToolMessage with stringified result (for LangGraph flow)
                    result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                    tool_message = ToolMessage(
                        content=result_str,
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    tool_results_messages.append(tool_message)
            
            except Exception as e:
                print(f"[tool_execution] Error executing {tool_name}: {e}")
                import traceback
                traceback.print_exc()
                
                error_result = {
                    "status": "error",
                    "message": str(e),
                    "sources": []
                }
                
                # Store error too
                global_tool_results.append({
                    'tool': tool_name,
                    'result': error_result,
                    'full_result': error_result
                })

                tool_message = ToolMessage(
                    content=str(error_result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_results_messages.append(tool_message)
        
        return {"messages": tool_results_messages}

    workflow = StateGraph(State)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_execution_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


# ============================================================
# Startup
# ============================================================
print("\nBUILDING AGENT AT STARTUP...")
app_agent = build_agent()
print("AGENT BUILD COMPLETE\n")


# ============================================================
# Gemini Fallback Handler
# ============================================================
def get_gemini_fallback(query: str) -> tuple[str, str]:
    """Call Gemini API directly when tools don't find answers."""
    print(f"[gemini_fallback] Calling Gemini for query: {query[:60]}")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY,
        )
        
        response = llm.invoke([
            SystemMessage(content="You are an expert agricultural assistant. Provide clear, detailed answers about agriculture, crops, pests, and farming practices."),
            HumanMessage(content=query)
        ])
        
        answer = response.content if hasattr(response, 'content') else str(response)
        print(f"[gemini_fallback] ✓ Got answer from Gemini ({len(answer)} chars)")
        return answer, "success"
    
    except Exception as e:
        print(f"[gemini_fallback] ✗ Error calling Gemini: {e}")
        return f"Unable to generate answer: {str(e)}", "error"


# ============================================================
# Source Extraction (Using Mohan's proven approach)
# ============================================================
def extract_sources_from_tool_results(tool_results: List[Dict[str, Any]]) -> List[str]:
    """
    Extract source filenames directly from RAW tool results.
    
    Handles multiple result formats:
    - Dict with 'sources' field
    - Dict with 'results' field
    - Direct list results (from tools like VignanUniversity)
    
    For list results, we report them as sources if they contain meaningful data.
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
        result_data = tool_result.get("full_result") or tool_result.get("result")
        
        if not result_data:
            print(f"[extract_sources] {tool_name}: No result data")
            continue
        
        # ✅ NEW: Handle list results directly (e.g., VignanUniversity returns list)
        if isinstance(result_data, list):
            if len(result_data) > 0:
                print(f"[extract_sources] {tool_name}: Got list with {len(result_data)} items")
                # For VignanUniversity: extract source/document from each item
                for item in result_data:
                    if isinstance(item, dict):
                        # Try different possible source field names
                        source = (item.get("source") or 
                                item.get("document") or 
                                item.get("filename") or 
                                item.get("pdf"))
                        if source:
                            source_str = str(source).strip()
                            if source_str:
                                sources.add(source_str)
                                print(f"    → {source_str}")
                        # If no explicit source field, try metadata
                        elif item.get("metadata"):
                            metadata = item["metadata"]
                            if isinstance(metadata, dict):
                                source = (metadata.get("source") or 
                                        metadata.get("document") or 
                                        metadata.get("filename"))
                                if source:
                                    sources.add(str(source).strip())
                                    print(f"    → {source}")
                # If no individual sources found, use tool name
                if len(sources) == 0:
                    sources.add(tool_name)
                    print(f"    → {tool_name} (no specific source found)")
            continue
        
        # Handle stringified JSON (fallback, but shouldn't be needed with global capture)
        if isinstance(result_data, str):
            print(f"[extract_sources] {tool_name}: Result is string, parsing JSON...")
            try:
                result_data = json.loads(result_data)
            except:
                print(f"[extract_sources] {tool_name}: Could not parse, skipping")
                continue
        
        if not isinstance(result_data, dict):
            continue
        
        print(f"[extract_sources] {tool_name}:")
        
        # Extract from 'sources' field
        if "sources" in result_data:
            src_list = result_data["sources"]
            if isinstance(src_list, list):
                print(f"  Found 'sources' with {len(src_list)} items")
                for src in src_list:
                    # Handle dict format: {'filename': 'file.pdf', ...}
                    if isinstance(src, dict) and "filename" in src:
                        filename = src["filename"]
                        if filename and isinstance(filename, str):
                            filename = filename.strip()
                            if filename:
                                sources.add(filename)
                                print(f"    → {filename}")
                    # Handle plain string format
                    elif isinstance(src, str) and src.strip():
                        sources.add(src.strip())
                        print(f"    → {src.strip()}")
        
        # Extract from 'results' field
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
    
    final_sources = sorted(list(sources))
    print(f"[extract_sources] FINAL: {final_sources}")
    return final_sources


def clean_response_text(text: str) -> str:
    """Clean response text by removing markdown formatting."""
    if not text:
        return ""
    
    import re
    
    cleaned = re.sub(r'```[\s\S]*?```', '', text)
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
    cleaned = re.sub(r'^#{1,6}\s+', '', cleaned)
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
    cleaned = re.sub(r'__([^_]+)__', r'\1', cleaned)
    cleaned = re.sub(r'_([^_]+)_', r'\1', cleaned)
    cleaned = cleaned.replace("\\n", "\n")
    
    if "📚 Sources:" in cleaned or "Sources:" in cleaned:
        if "📚 Sources:" in cleaned:
            cleaned = cleaned.split("📚 Sources:")[0]
        else:
            cleaned = cleaned.split("Sources:")[0]
    
    cleaned = cleaned.strip()
    return cleaned


def extract_final_answer(result: dict) -> str:
    """Extract final text answer from agent result."""
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


def has_meaningful_tool_results(tool_results: List[Dict[str, Any]]) -> bool:
    """Check if tool results contain meaningful information."""
    if not tool_results:
        print("[has_meaningful_tool_results] No tool results")
        return False
    
    for tool_result in tool_results:
        if not isinstance(tool_result, dict):
            continue
        
        result_data = tool_result.get("full_result") or tool_result.get("result")
        
        # ✅ NEW: Handle list results directly (some tools like VignanUniversity return lists)
        if isinstance(result_data, list):
            if len(result_data) > 0:
                print(f"[has_meaningful_tool_results] ✓ Found list result with {len(result_data)} items")
                return True
            continue
        
        if isinstance(result_data, str):
            try:
                result_data = json.loads(result_data)
            except:
                continue
        
        if not isinstance(result_data, dict):
            continue
        
        # Check for error status
        if result_data.get("status") == "error":
            continue
        
        # Check for 'sources'
        if result_data.get("sources") and isinstance(result_data["sources"], list):
            if len(result_data["sources"]) > 0:
                print(f"[has_meaningful_tool_results] ✓ Found sources")
                return True
        
        # Check for 'information'
        if result_data.get("information"):
            info_text = str(result_data["information"])
            if len(info_text) > 50:
                print(f"[has_meaningful_tool_results] ✓ Found information")
                return True
        
        # Check for 'results'
        if result_data.get("results") and isinstance(result_data["results"], list):
            if len(result_data["results"]) > 0:
                print(f"[has_meaningful_tool_results] ✓ Found results")
                return True
    
    print(f"[has_meaningful_tool_results] ✗ No meaningful results")
    return False


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="AgriGPT Agent")


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
            return {"status": "ok"}

        phone_number = message.get("from")
        user_message = message["text"].get("body", "").strip()
        if not phone_number or not user_message:
            return {"status": "ok"}

        print(f"[Webhook] Message from {phone_number}: {user_message}")

    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[Webhook] Parse error: {exc}")

    return {"status": "ok"}

@app.get("/hi", summary="Say Hi", tags=["Health"])
async def hi():
    """Returns a greeting from Claude."""
    return {"message": "Hi Claude !!"}


# ============================================================
# Chat Endpoint Models
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


# ============================================================
# MAIN CHAT ENDPOINT (Fixed)
# ============================================================
@app.post("/test/chat", response_model=ChatResponse)
def test_chat(request: ChatRequest):
    """
    Chat endpoint with TOOL-FIRST then GEMINI-FALLBACK strategy.
    
    FIX: Uses global_tool_results to capture RAW results for proper source extraction.
    """
    global global_tool_results
    
    print(f"\n[/test/chat] ========== START REQUEST ==========")
    print(f"[/test/chat] chatId={request.chatId} | phone={request.phone_number}")
    print(f"[/test/chat] message={request.message[:60]}")

    try:
        # ✅ Clear previous tool results for this request
        global_tool_results.clear()
        
        # Load history
        history = load_history(request.chatId)
        print(f"[/test/chat] Loaded {len(history)} messages from history.")
        
        system_prompt = SystemMessage(content="""You are AgriGPT, an expert agricultural assistant powered by a knowledge base of agricultural research and resources.

YOUR MISSION: Provide accurate, helpful answers using the knowledge base tools.

TOOL USAGE (MANDATORY):
1. Always call at least ONE tool before answering:
   - sme_divesh: Agricultural knowledge, AI in farming, best practices
   - pests_and_diseases: Crop diseases, pest identification, treatments
   - govt_schemes: Government agricultural programs and subsidies
   - VignanUniversity: Academic research and university resources

2. Use tool results as your PRIMARY source of information

3. If tools don't return relevant data, acknowledge this honestly

RESPONSE FORMATTING:
- Write in PLAIN TEXT only - NO markdown
- Be concise and helpful
- Mention which tool(s) provided your information""")
        
        history = [msg for msg in history if not isinstance(msg, SystemMessage)]
        history = [system_prompt] + history
        history.append(HumanMessage(content=request.message))
        
        # ========== STEP 1: Invoke agent ==========
        print("\n[STEP 1] Invoking agent...")
        result = app_agent.invoke({"messages": history})
        print(f"[STEP 1] Agent returned {len(result['messages'])} messages")
        
        final_answer = extract_final_answer(result)
        
        # Save history
        save_history(request.chatId, result["messages"], phone_number=request.phone_number)
        
        # ========== STEP 2: Check for meaningful results ==========
        print("\n[STEP 2] Checking tool results...")
        sources = []
        
        # ✅ Use global_tool_results which has RAW dicts
        has_meaningful = has_meaningful_tool_results(global_tool_results)
        print(f"[STEP 2] Has meaningful results: {has_meaningful}")
        
        # ========== STEP 3: Extract sources or use Gemini fallback ==========
        print("\n[STEP 3] Source strategy...")
        
        if has_meaningful:
            print("[STEP 3] ✅ Tools found results - extracting sources")
            # ✅ Pass RAW results to extraction
            sources = extract_sources_from_tool_results(global_tool_results)
            if not sources:
                sources = ["Knowledge Base"]
        else:
            print("[STEP 3] ❌ Tools found no results - using Gemini fallback")
            gemini_answer, gemini_status = get_gemini_fallback(request.message)
            
            if gemini_status == "success":
                final_answer = f"I couldn't find specific information in the knowledge base. Based on general agricultural knowledge:\n\n{gemini_answer}"
                sources = ["Gemini API"]
            else:
                final_answer = f"I couldn't retrieve information: {gemini_answer}"
                sources = ["Error - Unable to retrieve"]
        
        # ========== STEP 4: Clean response ==========
        cleaned_response = clean_response_text(final_answer)
        
        print(f"[STEP 4] FINAL SOURCES: {sources}")
        print(f"[/test/chat] ========== END REQUEST ==========\n")
        
        return ChatResponse(
            chatId=request.chatId,
            phone_number=request.phone_number,
            response=cleaned_response,
            sources=sources,
        )
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[/test/chat] ========== REQUEST FAILED ==========\n")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Production chat endpoint."""
    return test_chat(request)


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)