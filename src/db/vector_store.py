"""Thin wrapper over the raw chromadb client, replacing langchain_chroma.Chroma.

langchain-chroma pins chromadb>=1.3.5,<2.0.0, which conflicts with crewai's hard
pin of chromadb~=1.1.0 (true across every recent crewai release checked). This
project only ever used a small slice of langchain's Chroma interface
(similarity_search, similarity_search_with_score, add_texts), so it's cheaper
and safer to talk to chromadb directly than to keep both packages installed.
This reads/writes the exact same on-disk collection (./chroma_legal_data,
collection "company_legal_cases") and preserves the calling interface used by
src/agents/legal_agent.py and src/services/ingestion.py, so those files needed
no changes.
"""
import uuid


def open_persistent_client(path: str):
    """chromadb.PersistentClient(path=path), with a readable error instead of
    the raw Rust panic ("range start index N out of range for slice of
    length M") chromadb's sqlite bindings raise when `path` was written by an
    incompatible chromadb version (see this module's docstring / the
    chromadb note in requirements.txt — chromadb<1.2 cannot open a directory
    written by chromadb>=1.5, and the reverse can also fail across major
    versions)."""
    import chromadb

    try:
        return chromadb.PersistentClient(path=path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open the chromadb store at '{path}' "
            f"(chromadb {chromadb.__version__}). This usually means the "
            f"directory was written by a different chromadb version than "
            f"the one installed here. Fix: pip install \"chromadb>=1.5.9\" "
            f"in this environment (see requirements.txt), or delete/"
            f"re-ingest '{path}' if it's safe to rebuild. Original error: {exc}"
        ) from exc


class Document:
    """Minimal stand-in for langchain_core.documents.Document — just the two
    attributes this codebase actually reads (page_content, metadata)."""

    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class ChromaVectorStore:
    """Drop-in replacement for the subset of langchain_chroma.Chroma this
    project uses, backed directly by a chromadb collection."""

    def __init__(self, client, collection_name: str, embeddings):
        """
        :param client: a chromadb client (e.g. chromadb.PersistentClient)
        :param collection_name: name of the collection to use/create
        :param embeddings: object with embed_query(text) -> list[float] and
            embed_documents(texts) -> list[list[float]] (e.g.
            langchain_ollama.OllamaEmbeddings)
        """
        self.embeddings = embeddings
        self.collection = client.get_or_create_collection(name=collection_name)

    def similarity_search_with_score(self, query: str, k: int = 4, filter: dict = None):
        query_embedding = self.embeddings.embed_query(query)
        query_kwargs = {"query_embeddings": [query_embedding], "n_results": k}
        if filter:
            query_kwargs["where"] = filter
        results = self.collection.query(**query_kwargs)

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        docs_and_scores = []
        for doc_text, meta, distance in zip(documents, metadatas, distances):
            docs_and_scores.append((Document(doc_text, meta or {}), distance))
        return docs_and_scores

    def similarity_search(self, query: str, k: int = 4, filter: dict = None):
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter)]

    def add_texts(self, texts, metadatas=None):
        texts = list(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = self.embeddings.embed_documents(texts)
        add_kwargs = {"ids": ids, "documents": texts, "embeddings": embeddings}
        if metadatas is not None:
            add_kwargs["metadatas"] = list(metadatas)
        self.collection.add(**add_kwargs)
