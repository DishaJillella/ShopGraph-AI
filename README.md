
# ShopGraph AI 🛍️🤖

A stateful, multi-intent e-commerce customer support system built using **LangGraph**.

This project demonstrates production-style AI workflow orchestration with structured routing, vector search (pgvector), database integration, memory management, and human approval gates.

---

## 🚀 Features

- 🔎 Intent classification (pricing, order status, support, refund, cancel)
- 🗂 Structured SQL database lookups
- 🧠 Semantic vector search using pgvector
- 🔁 Multi-step workflow routing with LangGraph
- 🛑 Human approval gate for sensitive actions (refund/cancel)
- 💬 Conversation memory with summarization
- 🔄 Retry logic for robustness
- 🧩 Hybrid structured + semantic retrieval system

---

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

## 🛠 Tech Stack

- Python
- LangGraph
- OpenAI API
- PostgreSQL
- pgvector
- Psycopg
- Docker
