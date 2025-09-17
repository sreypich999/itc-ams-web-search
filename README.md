# ITC/AMS Search Engine

An AI-powered research assistant for the **Institute of Technology of Cambodia (ITC)** and its **Department of Applied Mathematics and Statistics (AMS)**. This intelligent system provides comprehensive, accurate, and secure access to institutional knowledge through advanced semantic search and conversational AI.

## Features

- **AI-Powered Conversations**: Natural language interactions using Google Gemini LLM
- **Semantic Search**: Advanced vector-based search through institutional documents
- **Knowledge Base**: Curated information from official ITC/AMS sources
- **Enterprise Security**: Multi-layer security with malicious query detection
- **Multilingual Support**: English and Khmer language processing
- **Persistent Chat**: Session-based conversation history
- **Responsive UI**: Modern web interface with real-time interactions



<video autoplay muted loop playsinline controls width="720">
  <source src="https://raw.githubusercontent.com/sreypich999/itc-ams-web-search/main/Recording_2025-07-29_174810.mp4" type="video/mp4">
  Your browser does not support the video tag. 
  <a href="https://github.com/sreypich999/itc-ams-web-search/releases">Download the demo</a>.
</video>



## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask App      │    │   LangChain     │
│   (HTML/CSS/JS) │◄──►│   (app.py)       │◄──►│   Agent         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Security      │    │   Knowledge     │
                       │   Layer         │    │   Base Tool     │
                       └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   Vector DB     │
                                               │   (ChromaDB)    │
                                               └─────────────────┘
```



Phase 1: Building the Knowledge Base (The "Scraping" Part)
This phase happens first, usually by running python scripts/initial_data_ingestion.py.

Step 1: Load Configuration

The script starts by reading config/api_keys.yaml.

It extracts the list of whitelisted domains under domain_filters -> itc and domain_filters -> ams.

Example: https://itc.edu.kh/**, https://en.wikipedia.org/wiki/Institute_of_Technology_of_Cambodia, etc.

Step 2: Initialize Components

The DocumentProcessor is initialized (with chunk_size=500, chunk_overlap=50).

The Embedder is created using the all-MiniLM-L6-v2 model.

The ChromaDB vector database is initialized, connected to the folder ./data/chroma_db.

Step 3: Scrape Content from Each URL

For each URL in the whitelist, the scrape_and_extract_content() method (document_processor.py) is called.

It uses the requests library with a custom User-Agent to fetch the HTML content.

It uses BeautifulSoup to parse the HTML.

It cleans the HTML by removing unwanted tags: ['script', 'style', 'header', 'footer', 'nav', 'aside', '.sidebar', '.ad-block']. This strips out navigation, ads, and scripts, leaving only the main content.

It extracts text from relevant tags: ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td'].

It follows links within the same domain (up to a max_depth of 2) to get more content, but only if they are relevant (e.g., it avoids links to PDFs, images, or login pages).

Step 4: Chunk the Text

The large amount of scraped text is too big for the AI model to handle at once.

The chunk_text() method splits the clean text into smaller segments.

Chunking Strategy: It splits by words (not characters). Each chunk is 500 words long, with an overlap of 50 words between chunks. This overlap is crucial to prevent losing the context of an idea that might be split between two chunks.

Each chunk is also given metadata (the source URL and the time it was ingested).

Step 5: Generate Embeddings and Store

Each text chunk is passed to the Embedder.

The SentenceTransformer model (all-MiniLM-L6-v2) converts the text chunk into a vector (a list of 384 numbers). This vector is a numerical representation of the chunk's semantic meaning.

The vector, the original text, and its metadata are stored together in the ChromaDB vector database.

The process repeats for every chunk from every URL.

Final Result of Phase 1: A powerful, searchable knowledge base where every piece of ITC/AMS information is stored as a vector, ready for semantic search.

Phase 2: Answering a User Query (The "Search" Part)
This phase starts when a user types a question into the web interface and hits enter. The diagram above shows how the two phases connect: the stored knowledge base is used to answer the user's query.

Step 1: The User Asks a Question

The user types a question (e.g., "What are the admission requirements?") in the web browser.

The browser sends this question to the Flask server (app.py) via a POST /search request.

Step 2: Security & Validation Check

The Flask server immediately runs the user's query through a security function (_is_malicious_query()).

It checks for patterns that indicate hacking attempts, prompts for system information, or injection attacks (e.g., words like system, database, export, password, special characters like ; or |).

If detected, it immediately blocks the query and returns a security error. If it passes, processing continues.

Step 3: Context and History Retrieval

The server checks the user's session cookie for a session_id.

It asks the ChatMemory class to retrieve the last 10 messages from the SQLite database for that session. This provides the conversation context (e.g., if the user's previous question was "What programs does ITC offer?", and the follow-up is "What are the admission requirements?", the AI knows "requirements" refers to "programs").

Step 4: Query Expansion (Optional)

The QueryExpander can be used to enhance the query.

It translates the query into English and Khmer to find relevant content in both languages.

It uses the synonyms.yaml file to generate variations of the query (e.g., "admission" -> "enrollment", "registration").

Step 5: Semantic Search

The expanded query is sent to the KnowledgeBaseTool.

The tool takes the query and converts it into a vector using the same Embedder from Phase 1.

This query vector is then sent to ChromaDB with the instruction: "Find the 25 text chunks in the database whose vectors are most similar (closest) to this query vector."

ChromaDB performs this lightning-fast mathematical comparison and returns the top results, along with their text, source URL, and a similarity score.

Step 6: Response Synthesis by the LLM

The LangGraph agent now has the user's original question, the conversation history, and the most relevant text chunks from the knowledge base.

It constructs a detailed prompt for the Gemini AI model. The prompt includes:

System Instructions: The strict rules from _create_agent_prompt() (only use knowledge base, never make up information, strict formatting rules).

Conversation History: The last few messages.

Retrieved Context: The text chunks found by the KnowledgeBaseTool.

User's Question: The original query.

The Gemini model is instructed to synthesize an answer based only on the provided context. It is forbidden from using its own knowledge.

It formats the answer into the strict JSON structure with answer, summary, details, list_items, and sources.

Step 7: Delivery and Display

The Flask server receives this JSON response from the agent.

It stores both the user's question and the AI's answer in the SQLite conversation history.

It sends the JSON response back to the user's web browser.

The browser's JavaScript renders the answer: it uses marked.js to convert the Markdown into beautiful HTML, displays the bullet points, and formats the source links.

Final Result for the User: A comprehensive, well-structured, and sourced answer to their question, derived entirely from the official ITC/AMS knowledge base, delivered in a clean and engaging chat interface.




## Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API Key


### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sreypich999/itc-ams-web-search.git
   cd itc-ams-web-search
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

4. **Configure API settings**
   ```bash
   # Update config/api_keys.yaml with your settings
   # Update config/synonyms.yaml for domain-specific terms
   ```

5. **Initialize the knowledge base**
   ```bash
   python scripts/initial_data_ingestion.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the interface**
   ```
   Open http://localhost:5000 in your browser
   ```

## Project Structure

```
itc-ams-search-engine/
├── app.py                          # Main Flask application
├── config/
│   ├── api_keys.yaml              # API configurations & domain filters
│   └── synonyms.yaml              # Domain-specific synonyms
├── agents/
│   └── langchain_agent_orchestrator.py  # Core AI agent logic
├── tools/
│   ├── search_tools.py            # Search utilities
│   ├── knowledge_base_tool.py     # Vector DB search tool
│   └── translation_tool.py        # Language processing
├── utils/
│   ├── memory.py                  # Chat history management
│   ├── document_processor.py      # Web scraping & text processing
│   └── query_expander.py          # Query enhancement
├── vector_store/
│   ├── embedder.py                # Text embedding models
│   └── chroma_db.py              # Vector database interface
├── data/
│   ├── chroma_db/                # Vector database storage
│   └── memory.db                 # Chat history database
├── scripts/
│   └── initial_data_ingestion.py  # Setup script
├── templates/
│   └── index.html                # Web interface
├── static/
│   └── style.css                 # Styling
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Core Methodology

### 1. LangGraph Agent Architecture

The system uses **LangGraph** to orchestrate a sophisticated AI workflow:

```python
# Core agent workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)      # Decision making
workflow.add_node("tool", tool_node)        # Tool execution
workflow.set_entry_point("agent")
```

**Agent State Management:**
- `input`: User query
- `chat_history`: Conversation context
- `agent_outcome`: Current decision (AgentAction/AgentFinish)
- `intermediate_steps`: Tool execution history
- `execution_plan`: Multi-step query strategy

### 2. Multi-Step Query Planning

```python
def _generate_plan(self, query: str, chat_history: list) -> Tuple[str, list]:
    # Analyzes user intent
    # Creates 1-4 step execution plan
    # Returns strategy and specific search steps
```

**Planning Process:**
1. Context analysis from chat history
2. Query complexity assessment
3. Multi-step search strategy generation
4. Domain-specific optimization

### 3. Vector-Based Knowledge Retrieval

```python
# Semantic search pipeline
query → embeddings → similarity_search → domain_filter → ranking → results
```

**Search Flow:**
1. **Query Expansion**: Synonyms + translations
2. **Embedding**: Convert to 384-dimension vectors
3. **Similarity Search**: ChromaDB cosine similarity
4. **Domain Filtering**: ITC/AMS source validation
5. **Ranking**: Distance-based result ordering

### 4. Security Architecture

**Multi-Layer Protection:**
```python
def _is_malicious_query(query: str) -> bool:
    # Pattern-based detection
    # Command injection prevention
    # System access blocking
```

**Security Features:**
- Query pattern analysis
- System command blocking
- Data access restrictions
- Security event logging
- Session isolation

## API Endpoints

### `/search` (POST)
**Primary search endpoint**
```javascript
// Request
fetch('/search', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams({query: 'What programs does ITC offer?'})
})
```

**Response Format:**
```json
{
    "answer": "## ITC Academic Programs\n\nITC offers comprehensive engineering and technology programs...",
    "summary": "**Key Takeaways:** 5-year engineering degrees, practical focus, industry partnerships",
    "details": "### Program Structure:\n* **Duration:** 5 years\n* **Focus:** Hands-on learning...",
    "list_items": ["Civil Engineering", "Electrical Engineering", "..."],
    "sources": ["https://itc.edu.kh/programs", "..."]
}
```

### `/history` (GET)
**Retrieve chat history**
```javascript
fetch('/history')  // Returns conversation history for current session
```

### `/clear_history` (POST)
**Clear conversation history**
```javascript
fetch('/clear_history', {method: 'POST'})
```

## Configuration

### Environment Variables (.env)
```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash
FLASK_SECRET_KEY=your_secret_key_here
```

### API Configuration (config/api_keys.yaml)
```yaml
# LLM Services
llm_services:
  gemini:
    api_key: "GEMINI_API_KEY"
    model: "gemini-2.5-flash"

# Domain Filters - Define your knowledge sources
domain_filters:
  itc:
    - "https://itc.edu.kh/**"
    - "https://en.wikipedia.org/wiki/Institute_of_Technology_of_Cambodia"
  ams:
    - "https://itc.edu.kh/home-ams/**"
```

### Synonyms Configuration (config/synonyms.yaml)
```yaml
itc:
  canonical: "Institute of Technology of Cambodia"
  variations:
    - "ITC"
    - "Techno"
    - "សាលាតិចណូ"
    
ams:
  canonical: "Department of Applied Mathematics and Statistics"
  variations:
    - "AMS"
    - "ReDA Lab"
```

## Data Flow

### 1. Initialization Flow
```
Start → Load Config → Initialize LLM → Setup Vector DB → Create Agent → Ready
```

### 2. Query Processing Flow
```
User Query → Security Check → Plan Generation → Tool Execution → Response Formatting → JSON Output
```

### 3. Knowledge Ingestion Flow
```
URLs → Web Scraping → Text Extraction → Chunking → Embedding → Vector Storage
```

## Key Components

### LangChain Agent Orchestrator
- **Purpose**: Central AI coordination
- **Technology**: LangGraph state machines
- **Features**: Multi-step planning, tool orchestration

### Knowledge Base Tool
- **Purpose**: Vector database search
- **Technology**: ChromaDB + SentenceTransformers
- **Features**: Semantic search, domain filtering

### Document Processor
- **Purpose**: Content ingestion
- **Technology**: BeautifulSoup + Requests
- **Features**: Web scraping, text chunking

### Security Layer
- **Purpose**: Threat protection
- **Technology**: Pattern matching + logging
- **Features**: Query filtering, access control

## Performance Metrics

- **Response Time**: < 3 seconds average
- **Accuracy**: 90%+ domain-relevant responses
- **Security**: 0 successful attacks logged
- **Uptime**: 99.9% availability target

## Security Features

### Query Sanitization
```python
malicious_patterns = [
    r'\b(?:download|export|dump)\b.*\b(?:database|knowledge base)\b',
    r'\b(?:system|file|source code|config|\.env|api key)\b',
    r'\b(?:hack|exploit|vulnerability|bypass)\b'
]
```

### Domain Validation
```python
def _is_itc_ams_domain(self, url: str) -> bool:
    # Validates all content sources
    # Ensures only approved domains
```

### Privacy Protection
```python
def _sanitize_content(self, content: str) -> str:
    # Removes sensitive information
    # Protects user privacy
```

## Technology Stack

- **Backend**: Python 3.8+, Flask
- **AI/ML**: LangChain, LangGraph, Google Gemini
- **Vector DB**: ChromaDB
- **Embeddings**: SentenceTransformers
- **Frontend**: HTML5, CSS3, JavaScript ES6
- **Database**: SQLite (chat history)
- **Security**: Custom pattern matching + logging


## Acknowledgments

- Institute of Technology of Cambodia (ITC)
- Department of Applied Mathematics and Statistics (AMS)
- LangChain community
- ChromaDB team
- Google Gemini AI

---

**Built with ❤️ for the ITC/AMS community**








