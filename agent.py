import os
import re
import time

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from pgvector import Vector
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DB_URL = os.getenv("DATABASE_URL")
EMBED_MODEL = "text-embedding-3-small"
_EMBEDDINGS_READY = False
MEMORY_MAX_ITEMS = 10
MEMORY_KEEP_LAST = 4
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 0.5



class State(TypedDict):
    user_input: str
    intent: str
    order_info: str
    shipping_info: str
    support_info: str
    response: str
    memory: list[str]
    approved: bool

#retry_call adds reliability by retrying failed operations with exponential backoff before raising an error.
def retry_call(fn, label: str):
    last_err = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            if attempt == RETRY_ATTEMPTS:
                break
            time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    raise RuntimeError(f"{label} failed after {RETRY_ATTEMPTS} attempts") from last_err


def get_db_conn():
    def _connect():
        conn = psycopg.connect(DB_URL)
        register_vector(conn)
        return conn

    return retry_call(_connect, "DB connect")


def extract_order_id(text: str):
    match = re.search(r"(?:order\s*#?\s*|#)(\d{3,})", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{3,})\b", text)
    if match:
        return match.group(1)
    return None


def extract_product_name(text: str):
    match = re.search(r"price of (.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip("?")
    match = re.search(r"how much (?:is|are) (.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip("?")
    return None


def add_memory(state: State, response_text: str, extra: dict | None = None):
    history = list(state.get("memory", []))
    history.append(f"USER: {state.get('user_input')}")
    history.append(f"ASSISTANT: {response_text.strip()}")
    history = compact_memory(history)
    updates = {"response": response_text, "memory": history}
    if extra:
        updates.update(extra)
    return updates


def get_memory_summary(state: State):
    history = state.get("memory", [])
    for item in history:
        if item.startswith("SUMMARY:"):
            return item.replace("SUMMARY:", "", 1).strip()
    return ""


def is_repeated_question(state: State):
    current = state.get("user_input", "").strip().lower()
    if not current:
        return False
    for item in state.get("memory", []):
        if item.startswith("USER:"):
            prev = item.replace("USER:", "", 1).strip().lower()
            if prev == current:
                return True
    return False


def generate_llm_response(state: State, context: str):
    memory_summary = get_memory_summary(state)
    repeated = is_repeated_question(state)
    instructions = (
        "You are a helpful e-commerce support agent. "
        "Use the provided context to answer the user. "
        "If the user is repeating a question, acknowledge the repeat and be concise. "
        "Do not invent facts beyond the context. "
        "Keep the response to 3-6 sentences."
    )
    prompt = [
        f"User: {state.get('user_input')}",
        f"Context:\n{context}",
    ]
    if memory_summary:
        prompt.append(f"Memory summary:\n{memory_summary}")
    if repeated:
        prompt.append("Note: The user is repeating the same question.")
    response = retry_call(
        lambda: client.responses.create(
            model="gpt-4o-mini",
            instructions=instructions,
            input="\n\n".join(prompt),
        ),
        "LLM response",
    )
    return response.output_text.strip()


def summarize_history(items: list[str]):
    prompt = (
        "Summarize this conversation history in 3-5 concise bullet points. "
        "Focus on user intent, order ids, and key outcomes. "
        "Keep it under 80 words. Return plain text."
    )
    response = retry_call(
        lambda: client.responses.create(
            model="gpt-4o-mini",
            instructions=prompt,
            input="\n".join(items),
        ),
        "Summarization",
    )
    return response.output_text.strip()


def compact_memory(history: list[str]):
    if len(history) <= MEMORY_MAX_ITEMS:
        return history
    recent = history[-MEMORY_KEEP_LAST:]
    older = history[:-MEMORY_KEEP_LAST]
    summary = summarize_history(older)
    return [f"SUMMARY: {summary}"] + recent


def embed_text(text: str):
    response = retry_call(
        lambda: client.embeddings.create(model=EMBED_MODEL, input=text),
        "Embeddings",
    )
    return Vector(response.data[0].embedding)


def ensure_note_embeddings():
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, note FROM order_notes WHERE embedding IS NULL")
            rows = cur.fetchall()
            for note_id, note in rows:
                embedding = embed_text(note)
                cur.execute(
                    "UPDATE order_notes SET embedding = %s WHERE id = %s",
                    (embedding, note_id),
                )
        conn.commit()


def order_lookup_tool(order_id: str):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, shipping_eta FROM orders WHERE order_id = %s",
                (order_id,),
            )
            row = cur.fetchone()

    if not row:
        return f"Order {order_id} not found"
    status, eta = row
    return f"Order {order_id} is {status}, ETA: {eta}"



def shipping_lookup_tool(order_id: str):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT shipping_eta FROM orders WHERE order_id = %s",
                (order_id,),
            )
            row = cur.fetchone()

    if not row:
        return f"Order {order_id} not found"
    (eta,) = row
    return f"Order {order_id} will arrive {eta}"


def pricing_tool(product: str):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT product_name, price, currency "
                "FROM products "
                "WHERE product_name ILIKE %s "
                "ORDER BY product_name ASC "
                "LIMIT 1",
                (f"%{product}%",),
            )
            row = cur.fetchone()

    if not row:
        return f"No pricing found for {product}"
    name, price, currency = row
    return f"{name} costs {currency} {price}"

def classify_intent(state: State):
    text = state["user_input"]
    text_lower = text.lower()

    if "cancel" in text_lower:
        return {"intent": "cancel"}
    if "refund" in text_lower:
        return {"intent": "refund"}

    response = retry_call(
        lambda: client.responses.create(
            model="gpt-4o-mini",
            instructions=(
                "You classify customer requests into one label. "
                "Return exactly one label from: pricing, order_status, support, refund, cancel. "
                "No extra text."
            ),
            input=text,
        ),
        "Intent classification",
    )

    intent = response.output_text.strip().lower()

    if intent not in {"pricing", "order_status", "support", "refund", "cancel"}:
        intent = "support"

    return {"intent": intent}



def pricing_node(state: State):
    product = extract_product_name(state["user_input"]) or "iPhone 15"
    result = pricing_tool(product)
    response_text = generate_llm_response(
        state,
        context=f"Pricing lookup result: {result}",
    )
    return add_memory(state, response_text)


def check_order_db(state: State):
    order_id = extract_order_id(state["user_input"])
    if not order_id:
        return {"order_info": "Order id not found in request"}
    result = order_lookup_tool(order_id)
    return {"order_info": result}


def check_shipping(state: State):
    order_id = extract_order_id(state["user_input"])
    if not order_id:
        return {"shipping_info": "Order id not found in request"}
    result = shipping_lookup_tool(order_id)
    return {"shipping_info": result}


def generate_response(state: State):
    response_text = generate_llm_response(
        state,
        context=(
            f"Order info: {state.get('order_info')}\n"
            f"Shipping info: {state.get('shipping_info')}"
        ),
    )
    return add_memory(state, response_text)


def route(state: State):
    if state["intent"] == "pricing":
        return "pricing_node"
    elif state["intent"] == "order_status":
        return "check_order_db"
    elif state["intent"] == "refund":
        return "refund_node"
    elif state["intent"] == "cancel":
        return "cancel_node"
    else:
        return "support_node"


def approval_node(state: State):
    requires_approval = state.get("intent") in {"refund", "cancel"}
    approved = state.get("approved", False)
    if not requires_approval:
        return {"approved": True}
    if approved:
        return {"approved": True}
    while True:
        choice = input(
            "This action is sensitive (refund/cancel). Proceed? [y/n/q]: "
        ).strip().lower()
        if choice in {"y", "yes"}:
            return {"approved": True}
        if choice in {"n", "no"}:
            response_text = "Action not approved. I will not proceed."
            return add_memory(state, response_text, extra={"approved": False})
        if choice in {"q", "quit"}:
            response_text = "Quit requested. I will not proceed."
            return add_memory(state, response_text, extra={"approved": False})


def route_after_approval(state: State):
    if state.get("approved"):
        return "route_intent"
    return "end"


def refund_node(state: State):
    response_text = generate_llm_response(
        state,
        context="Refund requested. Next step: confirm order id and reason.",
    )
    return add_memory(state, response_text)


def cancel_node(state: State):
    order_id = extract_order_id(state["user_input"])
    if not order_id:
        response_text = generate_llm_response(
            state,
            context="Cancellation requested, but no order id was provided. Ask for the order id.",
        )
        return add_memory(state, response_text)

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM order_notes WHERE order_id = %s", (order_id,))
            cur.execute("DELETE FROM orders WHERE order_id = %s", (order_id,))
            deleted = cur.rowcount
        conn.commit()

    if deleted:
        response_text = generate_llm_response(
            state,
            context=f"Order {order_id} was cancelled and removed from the system.",
        )
    else:
        response_text = generate_llm_response(
            state,
            context=f"Order {order_id} was not found. No cancellation was made.",
        )
    return add_memory(state, response_text)


def support_node(state: State):
    global _EMBEDDINGS_READY
    if not _EMBEDDINGS_READY:
        ensure_note_embeddings()
        _EMBEDDINGS_READY = True

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            embedding = embed_text(state["user_input"])
            cur.execute(
                "SELECT note FROM ("
                "  SELECT DISTINCT ON (note) note, embedding "
                "  FROM order_notes "
                "  WHERE embedding IS NOT NULL "
                "  ORDER BY note, embedding <-> %s"
                ") t ORDER BY embedding <-> %s LIMIT 3",
                (embedding, embedding),
            )
            notes = [row[0] for row in cur.fetchall()]

    if notes:
        response_text = generate_llm_response(
            state,
            context="Related notes:\n" + "\n".join(f"- {n}" for n in notes),
        )
        return add_memory(
            state,
            response_text,
            extra={"support_info": "\n".join(notes)},
        )
    response_text = generate_llm_response(
        state,
        context="No related notes were found. Escalate to support.",
    )
    return add_memory(state, response_text)


graph = StateGraph(State)

graph.add_node("classify", classify_intent)
graph.add_node("route_intent", lambda state: state)
graph.add_node("approval", approval_node)
graph.add_node("pricing_node", pricing_node)
graph.add_node("check_order_db", check_order_db)
graph.add_node("check_shipping", check_shipping)
graph.add_node("generate_response", generate_response)
graph.add_node("support_node", support_node)
graph.add_node("refund_node", refund_node)
graph.add_node("cancel_node", cancel_node)

graph.add_edge(START, "classify")
graph.add_edge("classify", "approval")

graph.add_conditional_edges(
    "approval",
    route_after_approval,
    {
        "route_intent": "route_intent",
        "end": END,
    },
)

graph.add_conditional_edges(
    "route_intent",
    route,
    {
        "pricing_node": "pricing_node",
        "check_order_db": "check_order_db",
        "support_node": "support_node",
        "refund_node": "refund_node",
        "cancel_node": "cancel_node",
    },
)

graph.add_edge("check_order_db", "check_shipping")
graph.add_edge("check_shipping", "generate_response")

graph.add_edge("generate_response", END)
graph.add_edge("pricing_node", END)
graph.add_edge("support_node", END)
graph.add_edge("refund_node", END)
graph.add_edge("cancel_node", END)

app = graph.compile()


if __name__ == "__main__":
    memory = []
    while True:
        user_input = input("Enter your query (or type 'quit' to exit): ").strip()
        if user_input.lower() in {"quit", "q", "exit"}:
            break
        if not user_input:
            continue
        result = app.invoke({"user_input": user_input, "memory": memory})
        memory = result.get("memory", memory)
        print("\nRESPONSE:")
        print(result["response"])

