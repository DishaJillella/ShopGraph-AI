# Complete LangGraph Use Case: AI Customer Support Agent with Tools, Routing, and Parallel Analysis

## Scenario: E-Commerce AI Support Agent

You’re building an AI agent for an online store like Amazon/Flipkart.

The agent must:

1. Understand customer request
2. Classify intent
3. Decide what to do next
4. Use tools if needed
5. Run parallel checks
6. Generate final response

---

## Example User Requests

```
"Where is my order #123?"
"Refund my order"
"What is price of iPhone 15?"
"Cancel my order"
```

---

# System Architecture Overview

```
                START
                  ↓
           classify_intent
                  ↓
        ┌─────────┼──────────┐
        ↓         ↓          ↓
    pricing    order      support
                ↓
        parallel execution
        ┌─────────┴─────────┐
        ↓                   ↓
  check_order_db      check_shipping_api
        └─────────┬─────────┘
                  ↓
           generate_response
                  ↓
                 END
```

---

# Concepts Covered

This one system includes:

✔ State
✔ Nodes
✔ Edges
✔ Conditional routing
✔ Tool calling
✔ Parallel execution
✔ Agent decision logic
✔ Workflow orchestration

Everything in LangGraph workflows-agents.

---

# Full Implementation Exercise

---

## Step 1: Install

```bash
pip install langgraph langchain langchain-openai
```

---

## Step 2: Define State

State is shared memory across nodes.

```python
from typing_extensions import TypedDict

class State(TypedDict):
    user_input: str
    intent: str
    order_info: str
    shipping_info: str
    response: str
```

---

## Step 3: Define Tools

These simulate external systems.

```python
def order_lookup_tool(order_id: str):
    return f"Order {order_id} is confirmed"


def shipping_lookup_tool(order_id: str):
    return f"Order {order_id} will arrive tomorrow"


def pricing_tool(product: str):
    return f"{product} costs ₹80,000"
```

---

## Step 4: Intent Classification Node

This decides workflow path.

```python
def classify_intent(state: State):

    text = state["user_input"].lower()

    if "price" in text:
        intent = "pricing"

    elif "order" in text:
        intent = "order_status"

    else:
        intent = "support"

    return {"intent": intent}
```

---

## Step 5: Pricing Node (Tool Usage)

```python
def pricing_node(state: State):

    result = pricing_tool("iPhone 15")

    return {
        "response": result
    }
```

---

## Step 6: Parallel Execution Nodes

Two checks run simultaneously.

### Order DB check

```python
def check_order_db(state: State):

    result = order_lookup_tool("123")

    return {
        "order_info": result
    }
```

---

### Shipping API check

```python
def check_shipping(state: State):

    result = shipping_lookup_tool("123")

    return {
        "shipping_info": result
    }
```

---

## Step 7: Response Generator Node

Combines results.

```python
def generate_response(state: State):

    return {
        "response": f"""
Order info: {state.get('order_info')}
Shipping info: {state.get('shipping_info')}
"""
    }
```

---

## Step 8: Conditional Router

Decides workflow dynamically.

```python
def route(state: State):

    if state["intent"] == "pricing":
        return "pricing_node"

    elif state["intent"] == "order_status":
        return "check_order_db"

    else:
        return "support_node"
```

---

## Step 9: Support Node

```python
def support_node(state: State):

    return {
        "response": "Support agent will contact you"
    }
```

---

## Step 10: Build Graph

This combines everything.

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)

# nodes
graph.add_node("classify", classify_intent)
graph.add_node("pricing_node", pricing_node)
graph.add_node("check_order_db", check_order_db)
graph.add_node("check_shipping", check_shipping)
graph.add_node("generate_response", generate_response)
graph.add_node("support_node", support_node)

# edges
graph.add_edge(START, "classify")

# conditional routing
graph.add_conditional_edges(
    "classify",
    route,
    {
        "pricing_node": "pricing_node",
        "check_order_db": "check_order_db",
        "support_node": "support_node"
    }
)

# parallel execution
graph.add_edge("check_order_db", "check_shipping")
graph.add_edge("check_shipping", "generate_response")

# ending
graph.add_edge("generate_response", END)
graph.add_edge("pricing_node", END)
graph.add_edge("support_node", END)

app = graph.compile()
```

---

## Step 11: Run the Agent

```python
result = app.invoke({
    "user_input": "Where is my order?"
})

print(result["response"])
```

---

# Expected Output

```
Order info: Order 123 is confirmed
Shipping info: Order 123 will arrive tomorrow
```

---

# What This One Exercise Teaches You

| Feature                 | Where Used                      |
| ----------------------- | ------------------------------- |
| State                   | State class                     |
| Workflow                | classify → response             |
| Conditional routing     | route()                         |
| Agent decision          | classify_intent                 |
| Tool usage              | pricing_tool, order_lookup_tool |
| Parallel execution      | check_order_db + check_shipping |
| Multi-step workflow     | full graph                      |
| Real-world agent design | full system                     |

---

# Real-World Systems Built Like This

This pattern is used in:

• ChatGPT tools
• Customer support bots
• AI assistants
• Autonomous agents
• RAG agents
• Multi-tool agents

---

# Final Challenge

Try adding:

• real LLM intent classification
• real database lookup
• vector search
• memory
• retries
• human-in-the-loop