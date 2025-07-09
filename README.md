# Tavily & LangGraph RAG Pipeline

This project implements a LangGraph-based RAG pipeline using the [Tavily](https://www.tavily.com/) API and Cohere LLMs. It parses user queries into subqueries, searches for relevant information using Tavily, and generates structured LLM responses via LangGraph state orchestration.


<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="Python" width="50" height="50"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vscode/vscode-original.svg" alt="VSCode" width="50" height="50"/>
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/jupyter.svg" alt="Jupyter" width="50" height="50"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" width="50" height="50"/>
  <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/1.53.0/files/light/langchain-color.png" alt="LangChain" width="50" height="50"/>
  <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/light/langgraph-color.png" alt="LangGraph" width="50" height="50"/>
  <img src="https://qjkcnuesiiqjpohzdjjm.supabase.co/storage/v1/object/public/aops_marketplace/logos/tavily.png" alt="Tavily" width="50" height="50"/>
  <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/light/cohere-color.png" alt="Cohere" width="50" height="50">
</p>

---

## 📌 Features

- 🔀 **Subquery Parsing:** Splits compound queries into multiple simpler subqueries using an LLM.
- 🔍 **Tavily Search:** Fetches relevant context for each subquery with dynamic depth selection (basic/advanced).
- 🧠 **LLM Answer Generation:** Generates final answers using Cohere’s `command-r` model.
- ♻️ **Checkpointing:** Uses LangGraph memory-based checkpointing to skip redundant steps in repeated queries.
- ✅ **Robust Testing:** Includes tests for various categories.

---

## 📁 Project Structure

```
tavilyHW/  
├── agent/  
│   ├── llm_response.py         # LLM wrapper (Cohere) 
│   ├── tavily_search.py        # Tavily search with retry, score filtering, and formatting  
│   └── langraph_pipeline.py    # LangGraph state machine  
├── util.py                     # Colored terminal printing  
├── demo.ipynb                  # Notebook with test cases  
├── requirements.txt            
└── README.md  
```

---

## Getting Started

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd tavilyHW
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Environment Variables

Before running the notebook, export your API keys:

```bash
export COHERE_API_KEY="your_cohere_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
```

### 3. Run the Notebook

Activate the virtual environment, then start Jupyter:

```bash
jupyter notebook demo.ipynb
```

### 4. Test Coverage

The notebook `demo.ipynb` demonstrates the capabilities and robustness of the LangGraph-based RAG pipeline using a variety of test cases:

- ✅ **Simple Queries**: Ensure the system handles straightforward prompts.
  - _Example_: `"What is LangGraph?"`

- 🔀  **Multi-Subquery Parsing**: Validate the LLM's ability to decompose complex prompts into atomic subqueries.
  - _Example_: `"What are the features of LangGraph and who uses it?"`

- 🔄 **Checkpointing**: Confirms that the graph avoids redundant work on repeated queries using in-memory checkpointing.
  - _Example_: Querying `"Where is France located?"` multiple times.

- 💥 **Stress Testing**: Evaluate system behavior on long, multi-part prompts.
  - _Example_: `"Tell me about LangGraph's benefits, use cases, integration with LLMs, industry adoption, competitors, and deployment options."`

- 🧪 **Robustness**: Assess response to malformed or nonsensical inputs.
  - _Example_: `"Blargle wib wib ahsheli LangGraph elephant?"`

- ✂️ **Short Prompts**: Check system’s ability to handle extremely short or vague prompts.
  - _Example_: `"LangGraph"`