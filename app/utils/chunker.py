"""
Smart chunking — parent-child model.

Parent (~1500 chars): stored in DB + Redis, returned to LLM as context.
Child  (~300 chars) : indexed in Qdrant for dense retrieval.
Each child stores parent_id in metadata.

Splitting strategy:
  - Markdown / DOCX with headings  → split by ## / ### sections first
  - Plain text / no headings       → RecursiveCharacterTextSplitter fallback
"""
from __future__ import annotations

import io
import logging
import re
import uuid
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

PARENT_SIZE    = 1500
PARENT_OVERLAP = 150
CHILD_SIZE     = 400
CHILD_OVERLAP  = 50


@dataclass
class ParentChunk:
    id:       str
    content:  str
    index:    int
    metadata: dict = field(default_factory=dict)


@dataclass
class ChildChunk:
    id:        str
    parent_id: str
    content:   str
    index:     int
    metadata:  dict = field(default_factory=dict)




def extract_text(file_bytes: bytes, mime_type: str | None) -> str:
    mime = (mime_type or "").lower()
    if "pdf" in mime:
        return _extract_pdf(file_bytes)
    if "wordprocessingml" in mime or "docx" in mime:
        return _extract_docx(file_bytes)
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="replace")


def _extract_pdf(data: bytes) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t.strip())
    return "\n\n".join(parts)


def _extract_docx(data: bytes) -> str:
    """Preserves heading hierarchy as Markdown ## / ### markers."""
    from docx import Document
    doc   = Document(io.BytesIO(data))
    lines = []
    for para in doc.paragraphs:
        if not para.text.strip():
            continue
        s = para.style.name or ""
        if "Heading 1" in s:
            lines.append(f"## {para.text.strip()}")
        elif "Heading 2" in s:
            lines.append(f"### {para.text.strip()}")
        elif "Heading 3" in s:
            lines.append(f"#### {para.text.strip()}")
        else:
            lines.append(para.text.strip())
    return "\n\n".join(lines)




def _recursive_split(text: str, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    sp = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=[r"\n\n", r"\n", r"(?<=[.!?])\s+", r" ", ""],
        is_separator_regex=True,
    )
    return [c for c in sp.split_text(text) if c.strip()]


def _split_by_headings(text: str) -> list[str]:
    parts = re.split(r"(?m)(?=^#{2,4}\s)", text)
    return [p.strip() for p in parts if re.match(r"^#{2,4}\s", p.strip())]


def _split_parents(text: str) -> list[str]:
    if re.search(r"(?m)^#{2,4}\s+\S", text):
        sections = _split_by_headings(text)
        parents  = []
        for s in sections:
            if len(s) <= PARENT_SIZE * 2:
                parents.append(s)
            else:
                parents.extend(_recursive_split(s, PARENT_SIZE, PARENT_OVERLAP))
        return [p for p in parents if p.strip()]
    return _recursive_split(text, PARENT_SIZE, PARENT_OVERLAP)




def build_parent_child_chunks(
    text: str,
    document_id: str,
    conversation_id: str,
    filename: str,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """
    Returns (parents, children).
    children[i].parent_id links back to a parent.
    Only children are embedded in the vector store.
    Parents are stored in DB (document_chunks) and Redis for fast lookup.
    """
    parent_texts = _split_parents(text)
    parents:  list[ParentChunk] = []
    children: list[ChildChunk]  = []
    child_idx = 0

    for p_idx, p_text in enumerate(parent_texts):
        p_id = str(uuid.uuid4())
        parents.append(ParentChunk(
            id=p_id, content=p_text, index=p_idx,
            metadata={
                "document_id":     document_id,
                "conversation_id": conversation_id,
                "filename":        filename,
                "chunk_type":      "parent",
                "parent_index":    p_idx,
            },
        ))
        for c_text in _recursive_split(p_text, CHILD_SIZE, CHILD_OVERLAP):
            children.append(ChildChunk(
                id=str(uuid.uuid4()), parent_id=p_id,
                content=c_text, index=child_idx,
                metadata={
                    "document_id":     document_id,
                    "conversation_id": conversation_id,
                    "filename":        filename,
                    "chunk_type":      "child",
                    "parent_id":       p_id,
                    "parent_index":    p_idx,
                    "child_index":     child_idx,
                },
            ))
            child_idx += 1

    log.info("Chunked", extra={"doc": document_id, "parents": len(parents), "children": len(children)})
    return parents, children


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Legacy simple splitter — kept for backward compat."""
    return _recursive_split(text, chunk_size, overlap)
