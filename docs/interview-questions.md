# RAGFlow Technical Interview Questions

> A comprehensive collection of technical interview questions based on the RAGFlow project, designed for full-stack developers with a focus on Agent technology.

## How to Use This Question Bank

- **Interview length suggestion**:
  - 30 minutes: pick 4-6 questions (1 architecture + 1 backend + 1 frontend + 1 RAG)
  - 60 minutes: pick 8-10 questions with 1 coding challenge
  - 90 minutes: include system design scenario + failure recovery deep dive
- **Difficulty tags**:
  - `L1` Fundamentals (concepts and terminology)
  - `L2` Practical implementation (trade-offs and hands-on decisions)
  - `L3` Advanced (scalability, reliability, multi-tenant, cost)
- **Evaluation dimensions**:
  - Architecture clarity and decomposition ability
  - Data model and consistency reasoning
  - API and engineering quality
  - Frontend interaction design and performance awareness
  - Reliability, observability, and production mindset
- **Scoring rubric (5-point per question)**:
  - `5`: complete, structured, with concrete implementation details and trade-offs
  - `4`: mostly complete, minor gaps in edge cases
  - `3`: core idea correct, lacks implementation depth
  - `2`: partial understanding, cannot connect to RAGFlow context
  - `1`: vague or incorrect explanation

## Table of Contents

- [How to Use This Question Bank](#how-to-use-this-question-bank)
- [System Architecture & Design](#system-architecture--design)
- [Agent System Deep Dive](#agent-system-deep-dive)
- [RAG Technology Core](#rag-technology-core)
- [Backend Development (Python/Flask)](#backend-development-pythonflask)
- [Frontend Development (React/TypeScript)](#frontend-development-reacttypescript)
- [Database & Storage](#database--storage)
- [Performance Optimization](#performance-optimization)
- [DevOps & Deployment](#devops--deployment)
- [Extended Deep Dive: Architecture, Data, Backend, Frontend](#extended-deep-dive-architecture-data-backend-frontend)
- [Coding Challenges](#coding-challenges)
- [Open-Ended Questions](#open-ended-questions)

---

## System Architecture & Design

### Q1: Describe RAGFlow's overall architecture

**What it tests**: System architecture understanding, tech stack knowledge

**Key points to cover**:
- **Frontend-Backend Separation**: React/TypeScript frontend + Flask Python backend
- **Microservices Architecture**: Orchestrated via Docker Compose
- **Core Services**:
  - API Server (Flask): Business logic
  - Document Processing: PDF parsing, OCR, layout analysis
  - Vector DB (Elasticsearch/Infinity): Vector retrieval
  - MySQL: Metadata storage
  - Redis: Caching and session management
  - MinIO: Object storage
- **Modular Design**: apps (routing) → services (business logic) → models (data layer)

**Follow-up**: How would you scale this architecture to handle 10x traffic?

---

### Q2: Why does RAGFlow support multiple vector databases (Elasticsearch, Infinity, OpenSearch)?

**What it tests**: Technology selection, trade-off analysis

**Key points to cover**:
- **Elasticsearch**: Mature ecosystem, full-text + vector search, production-ready
- **Infinity**: High-performance vector search, optimized for large-scale deployments
- **OpenSearch**: Open-source alternative to Elasticsearch, AWS compatibility
- **Design Pattern**: Strategy pattern for swappable implementations
- **Configuration**: Controlled via `DOC_ENGINE` environment variable and Docker profiles

**Follow-up**: How would you implement a new vector database adapter?

---

### Q3: How do you ensure data consistency across multiple data sources?

**What it tests**: Distributed systems, transaction handling

**Key points to cover**:
- **Eventual Consistency**: Acceptable for most RAG use cases
- **Transaction Boundaries**: MySQL transactions for critical metadata
- **Async Processing**: Document processing pipeline with retry mechanisms
- **Idempotency**: Ensuring operations can be safely retried
- **Monitoring**: Health checks and data validation

**Follow-up**: What happens if Elasticsearch indexing fails after MySQL commit?

---

## Agent System Deep Dive

### Q4: Explain the Agent workflow system architecture in RAGFlow

**What it tests**: Agent technology understanding, workflow orchestration

**Key points to cover**:
- **Component-Based Architecture**: Modular components (LLM, Retrieval, Categorize, etc.)
- **Canvas System**: Visual workflow builder in frontend
- **Execution Engine**: Graph-based execution with dependency resolution
- **Component Types**:
  - `Begin`: Entry point
  - `Retrieval`: Knowledge base search
  - `Generate`: LLM text generation
  - `Categorize`: Intent classification
  - `Answer`: Response formatting
  - `RelevantCheck`: Relevance scoring
- **Data Flow**: Context passing between components via JSON

**Follow-up**: How would you implement conditional branching in the workflow?

---

### Q5: How are Agent components designed to be pluggable and extensible?

**What it tests**: Design patterns, extensibility

**Key points to cover**:
- **Base Component Class**: Abstract interface for all components
- **Registration System**: Component registry for discovery
- **Template System**: Pre-built workflows in `agent/templates/`
- **Tool Integration**: External API wrappers (Tavily, Wikipedia, SQL)
- **Configuration Schema**: JSON schema for component parameters
- **Execution Context**: Shared state management

**Follow-up**: Design a new component for web scraping.

---

### Q6: How does RAGFlow handle tool calling in Agent workflows?

**What it tests**: LLM tool use, API integration

**Key points to cover**:
- **Tool Definitions**: Function schemas with parameters
- **LLM Integration**: Passing tool definitions to LLM
- **Execution Layer**: Safe execution environment (sandbox)
- **Error Handling**: Timeout, rate limiting, error recovery
- **Available Tools**:
  - Search (Tavily, DuckDuckGo)
  - Wikipedia
  - SQL execution
  - HTTP requests
  - Code execution (Python, Node.js)

**Follow-up**: How would you implement rate limiting for external API calls?

---

## RAG Technology Core

### Q7: Explain the RAG (Retrieval-Augmented Generation) pipeline in RAGFlow

**What it tests**: RAG fundamentals, implementation details

**Key points to cover**:
1. **Document Ingestion**: Upload → Parse → Chunk → Embed
2. **Retrieval Phase**:
   - Query embedding
   - Vector similarity search
   - Reranking (optional)
   - Top-k selection
3. **Generation Phase**:
   - Context assembly
   - Prompt construction
   - LLM generation
   - Response formatting
4. **Optimization**: Caching, batch processing, async operations

**Follow-up**: How do you handle multi-turn conversations in RAG?

---

### Q8: What chunking strategies does RAGFlow support and when to use each?

**What it tests**: Document processing, chunking algorithms

**Key points to cover**:
- **Naive Chunking**: Fixed character/token count
- **Semantic Chunking**: Sentence/paragraph boundaries
- **Hierarchical Chunking**: Parent-child relationships
- **Layout-Aware Chunking**: Respecting document structure
- **Trade-offs**:
  - Small chunks: Better precision, worse context
  - Large chunks: Better context, worse precision
- **Configuration**: `chunk_size`, `overlap`, `method`

**Follow-up**: How would you implement semantic chunking using embeddings?

---

### Q9: How does RAGFlow implement reranking and why is it important?

**What it tests**: Information retrieval, ranking algorithms

**Key points to cover**:
- **Two-Stage Retrieval**:
  1. Fast vector search (recall-oriented)
  2. Precise reranking (precision-oriented)
- **Reranking Models**: Cross-encoder models (BERT-based)
- **Benefits**: Improved relevance, better context selection
- **Implementation**: `rag/llm/rerank.py`
- **Trade-offs**: Latency vs. accuracy

**Follow-up**: Compare vector search vs. BM25 vs. hybrid search.

---

### Q10: What is Graph RAG and how is it implemented in RAGFlow?

**What it tests**: Advanced RAG techniques, graph databases

**Key points to cover**:
- **Concept**: Building knowledge graphs from documents
- **Entity Extraction**: NER + relationship extraction
- **Graph Construction**: Nodes (entities) + Edges (relationships)
- **Query Processing**: Graph traversal + vector search
- **Advantages**: Better for multi-hop reasoning, relationship queries
- **Implementation**: `rag/graphrag/` module

**Follow-up**: When would you choose Graph RAG over traditional RAG?

---

## Backend Development (Python/Flask)

### Q11: How is the Flask application structured in RAGFlow?

**What it tests**: Flask framework, application architecture

**Key points to cover**:
- **Blueprint Pattern**: Modular apps in `api/apps/`
  - `kb_app.py`: Knowledge base management
  - `dialog_app.py`: Chat/conversation
  - `document_app.py`: Document processing
  - `canvas_app.py`: Agent workflow canvas
  - `file_app.py`: File upload/management
- **Service Layer**: Business logic in `api/db/services/`
- **Models**: SQLAlchemy models in `api/db/db_models.py`
- **Entry Point**: `api/ragflow_server.py`

**Follow-up**: How would you add a new API endpoint for document summarization?

---

### Q12: How does RAGFlow handle large file uploads?

**What it tests**: File handling, streaming, performance

**Key points to cover**:
- **Streaming Upload**: Chunked transfer encoding
- **Size Limits**: `MAX_CONTENT_LENGTH` configuration
- **Storage Flow**: Upload → MinIO → Processing queue
- **Progress Tracking**: WebSocket or polling for status
- **Error Handling**: Partial upload recovery, validation
- **Security**: File type validation, virus scanning

**Follow-up**: How would you implement resumable uploads?

---

### Q13: Explain the LLM integration layer design

**What it tests**: Abstraction design, adapter pattern

**Key points to cover**:
- **Base Classes**: `rag/llm/chat_model.py`, `rag/llm/embedding_model.py`
- **Provider Adapters**: OpenAI, Azure, Anthropic, local models
- **Unified Interface**: Consistent API across providers
- **Configuration**: Model selection, parameters, API keys
- **Error Handling**: Retry logic, fallback models, rate limiting
- **Streaming**: SSE for real-time responses

**Follow-up**: How would you add support for a new LLM provider?

---

### Q14: How does document processing work in RAGFlow?

**What it tests**: Document parsing, OCR, layout analysis

**Key points to cover**:
- **Pipeline**: Upload → Parse → Layout Analysis → OCR → Chunking → Embedding
- **Parsers**: `deepdoc/` module
  - PDF: PyMuPDF, pdfplumber
  - Images: OCR (Tesseract, PaddleOCR)
  - Office: python-docx, openpyxl
- **Layout Analysis**: Table detection, figure extraction
- **Async Processing**: Celery/background tasks
- **GPU Acceleration**: Optional GPU mode for OCR

**Follow-up**: How would you optimize PDF parsing for 1000-page documents?

---

## Frontend Development (React/TypeScript)

### Q15: Why did RAGFlow choose UmiJS framework?

**What it tests**: Frontend framework selection

**Key points to cover**:
- **Convention over Configuration**: File-based routing
- **Plugin System**: Extensible architecture
- **Built-in Features**: State management, internationalization, SSR
- **Enterprise Ready**: Ant Design integration, TypeScript support
- **Developer Experience**: Fast refresh, mock data, proxy

**Follow-up**: Compare UmiJS vs. Next.js vs. Create React App.

---

### Q16: How is state management implemented in the frontend?

**What it tests**: State management patterns

**Key points to cover**:
- **Zustand**: Lightweight state management
- **Advantages over Redux**:
  - Less boilerplate
  - Better TypeScript support
  - Simpler API
  - No context provider needed
- **Store Organization**: Feature-based stores
- **Persistence**: LocalStorage integration
- **DevTools**: Redux DevTools compatibility

**Follow-up**: When would you choose Redux over Zustand?

---

### Q17: How is the Agent Canvas workflow builder implemented?

**What it tests**: Complex UI components, graph visualization

**Key points to cover**:
- **Graph Library**: React Flow or similar
- **Component Palette**: Drag-and-drop components
- **Connection Logic**: Edge validation, data flow
- **State Management**: Canvas state, undo/redo
- **Serialization**: JSON representation of workflows
- **Execution Visualization**: Real-time status updates

**Follow-up**: How would you implement undo/redo functionality?

---

### Q18: How does the frontend handle real-time chat updates?

**What it tests**: Real-time communication, streaming

**Key points to cover**:
- **SSE (Server-Sent Events)**: For LLM streaming responses
- **WebSocket**: For bidirectional communication
- **Optimistic Updates**: Immediate UI feedback
- **Message Queue**: Handling out-of-order messages
- **Error Recovery**: Reconnection logic

**Follow-up**: Compare SSE vs. WebSocket vs. Long Polling.

---

## Database & Storage

### Q19: How are Elasticsearch indices designed in RAGFlow?

**What it tests**: Search engine optimization, index design

**Key points to cover**:
- **Index Structure**:
  - Document chunks with embeddings
  - Metadata fields (doc_id, kb_id, user_id)
  - Vector field for similarity search
- **Mapping Configuration**: Field types, analyzers
- **Sharding Strategy**: Based on data volume
- **Query Optimization**: Filter before vector search
- **Index Lifecycle**: Creation, updates, deletion

**Follow-up**: How would you optimize for 100M+ documents?

---

### Q20: Explain the database schema design in MySQL

**What it tests**: Relational database design, normalization

**Key points to cover**:
- **Core Tables**:
  - `users`: User accounts
  - `knowledgebases`: KB metadata
  - `documents`: Document metadata
  - `chunks`: Document chunks (if not in ES)
  - `conversations`: Chat sessions
  - `messages`: Chat messages
  - `agents`: Agent configurations
- **Relationships**: Foreign keys, indexes
- **Optimization**: Proper indexing, query optimization

**Follow-up**: How would you implement multi-tenancy?

---

### Q21: How does MinIO object storage integrate with the system?

**What it tests**: Object storage, file management

**Key points to cover**:
- **Use Cases**: Original file storage, generated images
- **Bucket Organization**: Per-user or per-KB buckets
- **Access Control**: Pre-signed URLs, bucket policies
- **Integration**: S3-compatible API
- **Advantages**: Scalable, cost-effective, self-hosted

**Follow-up**: Compare MinIO vs. AWS S3 vs. local filesystem.

---

## Performance Optimization

### Q22: How would you optimize the document embedding process?

**What it tests**: Performance optimization, batch processing

**Key points to cover**:
- **Batch Processing**: `EMBEDDING_BATCH_SIZE` configuration
- **Async Processing**: Background workers
- **GPU Acceleration**: CUDA-enabled embedding models
- **Caching**: Embedding cache for duplicate content
- **Model Selection**: Smaller models for faster inference
- **Parallelization**: Multi-worker processing

**Follow-up**: Calculate the time to embed 10,000 documents.

---

### Q23: What caching strategies are used in RAGFlow?

**What it tests**: Caching design, Redis usage

**Key points to cover**:
- **Redis Caching**:
  - Query results cache
  - Embedding cache
  - Session cache
  - Rate limiting counters
- **Cache Invalidation**: TTL, manual invalidation
- **Cache Warming**: Pre-loading popular queries
- **Multi-Level Caching**: Memory + Redis

**Follow-up**: Design a cache invalidation strategy for document updates.

---

### Q24: How would you optimize vector search performance?

**What it tests**: Vector search optimization

**Key points to cover**:
- **Index Optimization**: HNSW, IVF parameters
- **Dimensionality Reduction**: PCA, quantization
- **Pre-filtering**: Metadata filters before vector search
- **Approximate Search**: Trade accuracy for speed
- **Hardware**: GPU acceleration, SSD storage
- **Sharding**: Distribute across multiple nodes

**Follow-up**: Explain HNSW algorithm and its parameters.

---

## DevOps & Deployment

### Q25: Explain Docker Compose profiles mechanism in RAGFlow

**What it tests**: Container orchestration, configuration management

**Key points to cover**:
- **Purpose**: Conditional service activation
- **Profiles**:
  - `elasticsearch`, `infinity`, `opensearch`: Vector DB selection
  - `cpu`, `gpu`: Hardware mode
  - `kibana`: Optional monitoring
  - `sandbox`: Code execution environment
- **Configuration**: `COMPOSE_PROFILES` environment variable
- **Benefits**: Single compose file, flexible deployment

**Follow-up**: How would you add a new optional service?

---

### Q26: What security considerations are important for production deployment?

**What it tests**: Security awareness, best practices

**Key points to cover**:
- **Credentials**: Change default passwords
- **Network**: Firewall rules, internal networks
- **SSL/TLS**: HTTPS for external access
- **Authentication**: JWT tokens, session management
- **Authorization**: RBAC, resource isolation
- **Input Validation**: SQL injection, XSS prevention
- **File Upload**: Type validation, size limits, virus scanning
- **API Rate Limiting**: Prevent abuse

**Follow-up**: How would you implement API rate limiting?

---

### Q27: How would you monitor RAGFlow in production?

**What it tests**: Observability, monitoring

**Key points to cover**:
- **Metrics**:
  - Request latency, throughput
  - Error rates
  - Resource usage (CPU, memory, disk)
  - Queue lengths
- **Logging**: Structured logging, log aggregation
- **Tracing**: Distributed tracing for requests
- **Alerting**: Threshold-based alerts
- **Tools**: Prometheus, Grafana, ELK stack
- **Health Checks**: Endpoint monitoring

**Follow-up**: Design a dashboard for RAGFlow operations.

---

## Extended Deep Dive: Architecture, Data, Backend, Frontend

> The following questions are intended to make interviews more detailed and scenario-driven. They are especially useful for senior full-stack or tech lead candidates.

### Architecture Questions

### AX1 (L2/L3): Walk through the full request path for "upload a PDF and ask a question"

**What it tests**: End-to-end architecture understanding, cross-service collaboration

**Key points to cover**:
- **Upload stage**: frontend upload request → backend validation → file persisted to MinIO
- **Parsing stage**: async task triggers parsing/OCR/layout analysis in `deepdoc/`
- **Indexing stage**: chunk generation + embedding + vector index write + MySQL metadata update
- **Query stage**: user question → retrieval (vector + optional rerank) → prompt assembly
- **Generation stage**: LLM answer generation with context and citations
- **Response stage**: streaming response to UI + trace/log update

**Strong answer indicators**:
- Can name sync vs async boundaries clearly
- Can explain where retries and idempotency are required
- Can identify likely bottlenecks (OCR, embedding, vector indexing)

**Follow-up**: If parsing succeeds but embedding fails, how do you expose status to users and recover safely?

---

### AX2 (L3): Design failure isolation and graceful degradation strategy

**What it tests**: Reliability engineering, architecture resilience

**Key points to cover**:
- **Dependency matrix**: what breaks when MySQL / Redis / ES / MinIO is unavailable
- **Graceful degradation**:
  - allow login/read-only when optional modules fail
  - disable only affected capabilities (e.g., indexing) instead of full outage
- **Circuit breaker + retry**: bounded retries with exponential backoff
- **Queue durability**: ensure pending jobs are resumable after restart
- **Operational playbook**: alerting, runbooks, and fallback modes

**Follow-up**: What should the frontend display when retrieval service is degraded but chat service is still available?

---

### AX3 (L3): How would you evolve from single-node Docker Compose to Kubernetes?

**What it tests**: Platform architecture migration planning

**Key points to cover**:
- **Separation of concerns**: stateless app pods vs stateful dependencies
- **Stateful workloads**: managed MySQL/ES/MinIO or StatefulSet strategy
- **Config and secrets**: env vars → ConfigMap/Secret, key rotation process
- **Autoscaling**: HPA based on CPU, queue length, and request latency
- **Release strategy**: rolling/canary + schema compatibility checks
- **Cost/perf trade-off**: GPU pool for OCR/embedding workloads

**Follow-up**: Which component would you migrate first with the lowest risk and highest benefit?

---

### Data Questions

### DX1 (L2/L3): Design data model for KB, document versions, and chunk lineage

**What it tests**: Data modeling, traceability, versioning design

**Key points to cover**:
- **Entity boundaries**: tenant, KB, document, document version, chunk, embedding record
- **Versioning strategy**:
  - immutable chunk snapshots per document version
  - active pointer to latest published version
- **Lineage fields**: source file hash, parser version, chunking params, embedding model id
- **Queryability**: support "which chunks were generated by parser X?"
- **Rollback**: ability to restore previous document index state

**Follow-up**: How do you prevent stale chunks from being retrieved after a document re-upload?

---

### DX2 (L3): Ensure consistency between MySQL metadata and vector index

**What it tests**: Distributed data consistency, reconciliation design

**Key points to cover**:
- **Write flow**: metadata first vs index first, and why
- **Outbox/event pattern**: reliable change propagation
- **Reconciliation job**: periodic scan to detect missing/orphan records
- **Idempotent workers**: deduplicate by document version + operation type
- **Recovery policy**: replay queue, dead-letter queue, manual repair tooling

**Follow-up**: Provide a reconciliation SQL + index-check strategy for nightly validation.

---

### DX3 (L2/L3): Build data lifecycle and retention policy

**What it tests**: Data governance, storage cost optimization

**Key points to cover**:
- **Hot/warm/cold tiers**: recent active KB vs archive storage
- **Retention rules**: chat logs, uploaded files, embeddings, temporary parse artifacts
- **Deletion semantics**: soft delete vs hard delete with delayed purge
- **Compliance**: tenant-level export/delete (GDPR-like requirements)
- **Backup/restore**: RPO/RTO objectives and regular restore drills

**Follow-up**: How do you design a secure "delete my data" workflow across MySQL, MinIO, and vector DB?

---

### Backend Questions

### BX1 (L2/L3): Design an idempotent asynchronous document ingestion API

**What it tests**: API design, backend reliability

**Key points to cover**:
- **API contract**: create ingestion job, query progress, retry/cancel job
- **Idempotency key**: file hash + tenant + parser config
- **State machine**: pending → parsing → chunking → embedding → indexed → failed
- **Failure details**: structured error codes and partial progress snapshot
- **Observability**: correlation IDs spanning API, queue worker, and indexing

**Follow-up**: How do you prevent duplicate billing/usage counting when a job is retried?

---

### BX2 (L3): Backend authorization model for multi-tenant RAG platform

**What it tests**: Security architecture, access control design

**Key points to cover**:
- **Role model**: org owner, admin, editor, viewer, API key service account
- **Resource scope**: tenant → KB → document → chat session permissions
- **Enforcement points**: middleware + service-layer guard (defense in depth)
- **Policy auditing**: who accessed which document and when
- **Token strategy**: short-lived access token + refresh + key rotation

**Follow-up**: How would you support temporary cross-tenant data sharing for collaboration?

---

### BX3 (L2/L3): Design backend observability for debugging wrong answers

**What it tests**: Troubleshooting methodology, production debugging

**Key points to cover**:
- **Trace chain**: question → retrieval candidates → rerank scores → final prompt → model response
- **Structured logs**: request IDs, tenant IDs, model IDs, latency per stage
- **Sampling policy**: full traces for failures, sampled traces for normal traffic
- **PII handling**: redact sensitive fields while preserving debugging value
- **SLOs**: latency, retrieval hit rate, answer groundedness proxy metrics

**Follow-up**: What minimal data should be persisted to reproduce a bad answer without storing full user content?

---

### Frontend Questions

### FX1 (L2/L3): Design frontend state architecture for chat + agent canvas

**What it tests**: Frontend architecture, state normalization

**Key points to cover**:
- **State split**:
  - server state (API data, async status)
  - client state (UI preferences, transient edits)
- **Store boundaries**: chat state, KB state, canvas graph state, user/session state
- **Normalization**: node/edge maps for efficient graph updates
- **Persistence**: draft workflow autosave and local crash recovery
- **Conflict handling**: optimistic updates + rollback on server rejection

**Follow-up**: How do you avoid stale state bugs when switching tenants quickly?

---

### FX2 (L2/L3): Build robust streaming chat UX under unstable networks

**What it tests**: Realtime UX, error recovery, user experience quality

**Key points to cover**:
- **Streaming protocol**: SSE chunk handling and incremental rendering
- **Resilience**: reconnect with resume token / message cursor
- **UI behavior**: typing indicator, partial response markers, retry affordance
- **Consistency**: deduplicate repeated chunks after reconnect
- **Backpressure**: throttle rendering for long outputs to keep UI responsive

**Follow-up**: How do you guarantee message ordering when user sends new input while previous response is still streaming?

---

### FX3 (L2/L3): Frontend performance strategy for large workflows and long chats

**What it tests**: Rendering performance, scalability in browser

**Key points to cover**:
- **Virtualization**: long message list/windowing
- **Memoization**: avoid unnecessary node re-renders in canvas
- **Chunked rendering**: progressively render heavy graph updates
- **Web worker usage**: offload expensive layout/transform tasks
- **Profiling**: use React Profiler and performance marks for bottleneck discovery

**Follow-up**: What specific metrics would you track to prevent performance regressions in CI?

---

### FX4 (L2/L3): Frontend security for enterprise RAG applications

**What it tests**: Web security awareness in product context

**Key points to cover**:
- **XSS protection**: sanitize rich content and markdown rendering
- **Secrets safety**: never expose provider API keys in browser bundles
- **Upload safety**: MIME validation and strict client-side checks as first layer
- **Session security**: CSRF strategy, token storage policy, logout invalidation
- **Tenant isolation in UI**: ensure all API calls carry tenant context and are server-validated

**Follow-up**: How would you design a secure "share conversation" feature with expiring links?

---

## Coding Challenges

### Q28: Implement a document chunking algorithm

**What it tests**: Algorithm design, coding ability

**Requirements**:
- Support fixed-size chunking with overlap
- Support semantic chunking (sentence boundaries)
- Handle edge cases (empty documents, very long sentences)

```python
def chunk_document(text: str, chunk_size: int, overlap: int, method: str) -> List[str]:
    """
    Chunk a document into smaller pieces.

    Args:
        text: Input document text
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters
        method: 'fixed' or 'semantic'

    Returns:
        List of text chunks
    """
    # Your implementation here
    pass
```

---

### Q29: Implement an LRU cache for embedding results

**What it tests**: Data structures, caching

**Requirements**:
- O(1) get and put operations
- Thread-safe
- Size limit
- TTL support (optional)

```python
class EmbeddingCache:
    def __init__(self, capacity: int):
        """Initialize LRU cache with given capacity."""
        pass

    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        pass

    def put(self, text: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        pass
```

---

### Q30: Design a rate limiter for LLM API calls

**What it tests**: Concurrency control, rate limiting

**Requirements**:
- Token bucket or sliding window algorithm
- Per-user rate limiting
- Distributed (Redis-based)
- Graceful degradation

```python
class RateLimiter:
    def __init__(self, redis_client, max_requests: int, window_seconds: int):
        """Initialize rate limiter."""
        pass

    def allow_request(self, user_id: str) -> bool:
        """Check if request is allowed."""
        pass

    def get_remaining(self, user_id: str) -> int:
        """Get remaining requests in current window."""
        pass
```

---

### Q31: Implement a simple Agent workflow executor

**What it tests**: Graph algorithms, workflow orchestration

**Requirements**:
- Execute components in dependency order
- Handle conditional branching
- Support parallel execution
- Error handling and rollback

```python
class WorkflowExecutor:
    def __init__(self, workflow: Dict):
        """Initialize with workflow definition."""
        pass

    def execute(self, input_data: Dict) -> Dict:
        """Execute workflow and return results."""
        pass

    def validate(self) -> bool:
        """Validate workflow structure."""
        pass
```

---

## Open-Ended Questions

### Q32: How would you add multi-modal support (images, audio) to RAGFlow?

**What it tests**: System extension, innovation

**Discussion points**:
- Multi-modal embedding models (CLIP, ImageBind)
- Storage considerations
- Query interface changes
- Use cases and benefits

---

### Q33: Design a multi-tenant architecture for RAGFlow

**What it tests**: Enterprise architecture, scalability

**Discussion points**:
- Data isolation strategies
- Resource quotas and billing
- Performance isolation
- Security considerations

---

### Q34: How would you implement A/B testing for different RAG strategies?

**What it tests**: Experimentation, metrics

**Discussion points**:
- Experiment framework
- Metrics to track (relevance, latency, cost)
- Traffic splitting
- Statistical significance

---

### Q35: What are the future trends in RAG technology?

**What it tests**: Industry knowledge, vision

**Discussion points**:
- Agentic RAG
- Multi-modal RAG
- Smaller, faster models
- Better evaluation metrics
- Integration with knowledge graphs

---

## Preparation Tips

### For Full-Stack Developers

1. **Backend Focus**:
   - Understand Flask blueprints and service layer
   - Study document processing pipeline
   - Review LLM integration patterns

2. **Frontend Focus**:
   - Familiarize with UmiJS and Zustand
   - Understand Canvas workflow implementation
   - Review real-time communication patterns

3. **Agent Technology**:
   - Deep dive into `agent/` directory
   - Understand component architecture
   - Study workflow execution engine

### Key Directories to Review

- `api/apps/`: API endpoints and routing
- `api/db/services/`: Business logic
- `rag/`: Core RAG implementation
- `agent/`: Agent system
- `web/src/`: Frontend code
- `docker/`: Deployment configuration

### Hands-On Practice

1. Set up local development environment
2. Add a new Agent component
3. Implement a custom chunking strategy
4. Build a simple RAG pipeline from scratch
5. Optimize a slow query

---

## Additional Resources

- [RAGFlow Documentation](https://ragflow.io/docs)
- [RAG Papers](https://github.com/Hannibal046/Awesome-LLM#rag)
- [Agent Frameworks](https://github.com/e2b-dev/awesome-ai-agents)
- [Vector Databases Comparison](https://github.com/erikbern/ann-benchmarks)

---

*Last Updated: 2026-02-09*
