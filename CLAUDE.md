# CLAUDE.md - Course Materials RAG System Guide

## Essential Development Commands

### Quick Start
```bash
# Start the application (run from project root)
./run.sh
```

### Manual Start
```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Environment Setup
```bash
# Install dependencies
uv sync

# Required .env file in root
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### Application URLs
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Dependency Management Notes
- Use `uv` to run Python files or add any dependencies

## High-Level Architecture

### Core Components Flow
```
User Query → FastAPI → RAGSystem → [VectorStore, AIGenerator] → Response
                   ↓
               SessionManager (context)
                   ↓
               ToolManager (search tools)
```

### Key Architecture Patterns

#### 1. Tool-Based RAG System
- **NOT traditional retrieve-then-generate**: Uses Anthropic's tool calling
- AI decides when to search via `CourseSearchTool`
- Single search per query maximum to prevent loops
- Sources tracked separately from AI response

#### 2. Dual Vector Collections
- `course_catalog`: Course metadata (titles, instructors, links)
- `course_content`: Actual course material chunks
- Semantic course name resolution via vector search

#### 3. Chunk Processing Strategy
- Sentence-based chunking (not fixed character splits)
- Context injection: "Course [title] Lesson [number] content: [chunk]"
- Chunk size: 800 chars, overlap: 100 chars
- Each chunk tagged with course_title and lesson_number

## Critical Technical Details

### Document Format Expected
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [lesson_url]
[lesson content...]

Lesson 1: [title]
[lesson content...]
```

### Vector Store Search Flow
1. Course name resolution (if provided) via semantic search of `course_catalog`
2. Content search in `course_content` with metadata filters
3. Results formatted with course/lesson context headers

### AI Generator Behavior
- Model: `claude-sonnet-4-20250514`
- Temperature: 0 (deterministic)
- Max tokens: 800
- System prompt emphasizes: brief, concise, educational responses
- Tool use limited to one search per query

### Configuration Centralization
All settings in `/backend/config.py`:
- Chunk settings, model parameters, database paths
- Environment variables loaded via dotenv

## Key Code Relationships

### Component Dependencies
- `RAGSystem` orchestrates all components
- `VectorStore` handles ChromaDB operations and search logic
- `DocumentProcessor` parses course documents and creates chunks
- `AIGenerator` manages Anthropic API with tool execution
- `SessionManager` maintains conversation history (max 2 exchanges)

### Tool Architecture
- Abstract `Tool` base class for extensibility
- `CourseSearchTool` implements semantic search with metadata filtering
- `ToolManager` handles tool registration and execution
- Tools can track sources independently (e.g., `last_sources`)

### Frontend-Backend Integration
- Vanilla JS frontend with marked.js for markdown rendering
- Session-based conversations with automatic session creation
- Real-time course statistics via `/api/courses` endpoint
- Source display via collapsible UI elements

## Development Patterns

### Adding New Course Documents
- Place `.txt` files in `/docs` folder
- Follow expected document format (see above)
- Documents processed automatically on startup
- Existing courses skipped to prevent duplicates

### Extending Search Capabilities
- Create new tool implementing `Tool` interface
- Register with `ToolManager` in `RAGSystem.__init__`
- Add tool definition following Anthropic's schema
- Implement source tracking if needed

### Modifying Chunk Strategy
- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in config
- Modify `DocumentProcessor.chunk_text()` for different splitting logic
- Update context injection in `process_course_document()`

### Database Persistence
- ChromaDB stores data in `/backend/chroma_db`
- Collections persist between restarts
- Use `clear_existing=True` in `add_course_folder()` for fresh rebuilds

## Common Issues & Solutions

### Empty Search Results
- Check course name spelling (uses semantic matching)
- Verify lesson numbers exist in course
- Ensure documents were processed correctly on startup

### ChromaDB Corruption
- Delete `/backend/chroma_db` folder
- Restart application to rebuild from `/docs` files

### AI Response Quality
- Modify system prompt in `AIGenerator.SYSTEM_PROMPT`
- Adjust temperature/max_tokens in `base_params`
- Review tool definitions for search specificity

### Session Management
- Sessions auto-created if not provided
- History limited to 2 exchanges by default
- Modify `MAX_HISTORY` in config for longer conversations

## File Structure Priority

When making changes, understand these file relationships:
- `app.py`: Entry point, route definitions, startup document loading
- `rag_system.py`: Main orchestrator - modify for new component integration
- `vector_store.py`: Core search logic - critical for result quality
- `ai_generator.py`: Response generation and tool execution
- `search_tools.py`: Tool definitions and search implementation
- `config.py`: All configuration settings
- `models.py`: Data structures - modify for new metadata fields

## Performance Considerations

- One tool call maximum per query prevents loops
- Sentence-based chunking preserves context better than fixed splits
- Vector search happens at course resolution AND content search
- Conversation history limited to prevent token bloat
- Static files served with no-cache headers in development mode