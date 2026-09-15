# Orivory API Documentation v2.0

**RAG-native Answer Engine for Researchers**

Base URL: `https://api.orivory.io/api/v1`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Authentication](#2-authentication)
3. [Conversations & Chat (RAG Core)](#3-conversations--chat-rag-core)
4. [Memory (Second Brain)](#4-memory-second-brain)
5. [Knowledge Graph](#5-knowledge-graph)
6. [Sources & Connectors](#6-sources--connectors)
7. [Feedback & Calibration (NEW)](#7-feedback--calibration-new)
8. [Insights (NEW)](#8-insights-new)
9. [Admin API](#9-admin-api)
10. [Webhooks (Future)](#10-webhooks-future)
11. [Rate Limits & Quotas](#11-rate-limits--quotas)
12. [Error Reference](#12-error-reference)
13. [Agent Clients & MCP Hub](#13-agent-clients--mcp-hub)
14. [Erasure Receipts](#14-erasure-receipts)
15. [Import Paths](#15-import-paths)
16. [Appendix A: OpenAPI Schema (YAML)](#appendix-a-openapi-schema-yaml)
17. [Appendix B: SDK Examples](#appendix-b-sdk-examples)

---

## 1. Overview

### Base URL

```
https://api.orivory.io/api/v1
```

### Authentication

All API requests (except `/auth/*`) require a Bearer token in the Authorization header:

```
Authorization: Bearer <access_token>
```

Tokens are obtained via the `/auth/login` or `/auth/register` endpoints. Access tokens expire after **1 hour**. Use `/auth/refresh` to obtain a new access token.

#### Token Response

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

#### Refreshing Tokens

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response:** Same structure as token response above.

---

### Rate Limiting

Rate limits are enforced per user per endpoint tier.

| Tier      | Requests/Minute | Requests/Day |
|-----------|----------------|--------------|
| Free      | 60             | 1,000        |
| Pro       | 300            | 10,000       |
| Enterprise| 1,000          | 100,000      |

Rate limit headers are returned on every response:

```http
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 299
X-RateLimit-Reset: 1704067200
```

When a limit is exceeded, the API returns `429 Too Many Requests` with a `Retry-After` header.

---

### Error Codes

All errors follow a consistent JSON structure:

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested conversation does not exist.",
    "details": {
      "resource_type": "conversation",
      "resource_id": "abc123"
    },
    "request_id": "req_abc123xyz"
  }
}
```

| HTTP Status | Error Code                | Description                           |
|-------------|---------------------------|---------------------------------------|
| 400         | `VALIDATION_ERROR`        | Invalid request parameters            |
| 401         | `UNAUTHORIZED`            | Missing or invalid token              |
| 403         | `FORBIDDEN`               | Insufficient permissions              |
| 404         | `RESOURCE_NOT_FOUND`      | Resource does not exist               |
| 409         | `CONFLICT`                | Resource already exists               |
| 422         | `UNPROCESSABLE_ENTITY`    | Semantically invalid input            |
| 429         | `RATE_LIMIT_EXCEEDED`     | Too many requests                     |
| 500         | `INTERNAL_ERROR`          | Server error                          |
| 503         | `SERVICE_UNAVAILABLE`     | Maintenance or overload               |

---

### Pagination

List endpoints support cursor-based pagination:

```http
GET /api/v1/chat/conversations?limit=20&cursor=eyJpZCI6MTIzfQ
```

**Query Parameters:**

| Parameter | Type    | Default | Max | Description                          |
|-----------|---------|---------|-----|--------------------------------------|
| `limit`   | integer | 20      | 100 | Number of items per page             |
| `cursor`  | string  | null    | —   | Opaque cursor for next page          |
| `sort`    | string  | `desc`  | —   | Sort order: `asc` or `desc`          |
| `order_by`| string  | varies  | —   | Field to sort by (endpoint-specific) |

**Pagination Response Headers:**

```http
X-Pagination-HasMore: true
X-Pagination-NextCursor: eyJpZCI6MTQzfQ
X-Pagination-TotalCount: 247
```

---

### Versioning

The API is versioned via the URL path (`/api/v1`). When a breaking change is introduced, a new version (`/api/v2`) is released with a 12-month deprecation window for the previous version.

Non-breaking additions (new optional fields, new endpoints) are added to the current version without version bumps.

---

## 2. Authentication

### POST /api/v1/auth/register

Register a new user account.

**Request:**

```json
{
  "email": "researcher@university.edu",
  "password": "SecureP@ssw0rd!",
  "full_name": "Dr. Jane Smith",
  "research_focus": "computational biology",
  "accept_terms": true
}
```

| Field           | Type    | Required | Description                          |
|-----------------|---------|----------|--------------------------------------|
| `email`         | string  | Yes      | Valid email address (unique)         |
| `password`      | string  | Yes      | Min 8 chars, 1 uppercase, 1 number |
| `full_name`     | string  | Yes      | Display name                         |
| `research_focus`| string  | No       | Primary research domain              |
| `accept_terms`  | boolean | Yes      | Must be `true`                       |

**Response `201 Created`:**

```json
{
  "user": {
    "id": "usr_a1b2c3d4",
    "email": "researcher@university.edu",
    "full_name": "Dr. Jane Smith",
    "research_focus": "computational biology",
    "created_at": "2025-01-15T10:30:00Z",
    "plan": "free"
  },
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

---

### POST /api/v1/auth/login

Authenticate and obtain tokens.

**Request:**

```json
{
  "email": "researcher@university.edu",
  "password": "SecureP@ssw0rd!"
}
```

**Response `200 OK`:**

```json
{
  "user": {
    "id": "usr_a1b2c3d4",
    "email": "researcher@university.edu",
    "full_name": "Dr. Jane Smith",
    "plan": "pro",
    "mfa_enabled": false
  },
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600
}
```

**Response `401 Unauthorized` (invalid credentials):**

```json
{
  "error": {
    "code": "INVALID_CREDENTIALS",
    "message": "Email or password is incorrect.",
    "request_id": "req_xyz789"
  }
}
```

**Response `401 Unauthorized` (MFA required):**

```json
{
  "error": {
    "code": "MFA_REQUIRED",
    "message": "Multi-factor authentication is required.",
    "details": {
      "mfa_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    }
  }
}
```

---

### POST /api/v1/auth/refresh

Obtain a new access token using a refresh token.

**Request:**

```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response `200 OK`:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

**Response `401 Unauthorized` (token expired or revoked):**

```json
{
  "error": {
    "code": "TOKEN_EXPIRED",
    "message": "The refresh token has expired. Please log in again."
  }
}
```

---

### POST /api/v1/auth/logout

Revoke the current refresh token and invalidate the session.

**Request:**

```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response `204 No Content`**

---

### GET /api/v1/auth/google/authorize

Initiate Google OAuth flow. Redirects the user to Google's consent screen.

**Query Parameters:**

| Parameter | Type   | Required | Description                          |
|-----------|--------|----------|--------------------------------------|
| `redirect_uri` | string | Yes   | OAuth callback URL                   |
| `state`   | string | Yes      | CSRF protection token                |

**Response:** Redirect to Google consent screen.

After consent, Google redirects to your `redirect_uri` with:

```
https://your-app.com/callback?code=4/0Adeu5B...&state=csrf_token
```

Exchange the code via `/api/v1/auth/google/callback`:

---

### POST /api/v1/auth/google/callback

Exchange a Google OAuth code for Orivory tokens.

**Request:**

```json
{
  "code": "4/0Adeu5B...",
  "redirect_uri": "https://your-app.com/callback"
}
```

**Response `200 OK`:**

```json
{
  "user": {
    "id": "usr_a1b2c3d4",
    "email": "researcher@gmail.com",
    "full_name": "Jane Smith",
    "oauth_provider": "google"
  },
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

---

### POST /api/v1/auth/forgot-password

Initiate password reset flow. Sends an email with a reset link.

**Request:**

```json
{
  "email": "researcher@university.edu"
}
```

**Response `200 OK`:**

```json
{
  "message": "If an account with that email exists, a password reset link has been sent."
}
```

---

### POST /api/v1/auth/reset-password

Reset password using a token from the forgot-password email.

**Request:**

```json
{
  "reset_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "new_password": "NewSecureP@ss123!"
}
```

**Response `200 OK`:**

```json
{
  "message": "Password has been reset successfully."
}
```

---

### GET /api/v1/auth/me

Get the currently authenticated user's profile.

**Response `200 OK`:**

```json
{
  "id": "usr_a1b2c3d4",
  "email": "researcher@university.edu",
  "full_name": "Dr. Jane Smith",
  "research_focus": "computational biology",
  "plan": "pro",
  "mfa_enabled": false,
  "created_at": "2025-01-15T10:30:00Z",
  "last_login": "2025-01-20T14:22:00Z",
  "settings": {
    "default_model": "claude-3-5-sonnet",
    "email_notifications": true,
    "digest_frequency": "daily"
  }
}
```

---

### PATCH /api/v1/users/me/change-password

Change the authenticated user's password.

**Request:**

```json
{
  "current_password": "OldSecureP@ss!",
  "new_password": "NewSecureP@ss123!"
}
```

**Response `200 OK`:**

```json
{
  "message": "Password changed successfully."
}
```

---

## 3. Conversations & Chat (RAG Core)

### GET /api/v1/chat/conversations

List all conversations for the authenticated user.

**Query Parameters:**

| Parameter    | Type    | Default | Description                        |
|--------------|---------|---------|-----------------------------------|
| `limit`      | integer | 20      | Items per page (max 100)           |
| `cursor`     | string  | null    | Pagination cursor                  |
| `sort`       | string  | `desc`  | Sort by `updated_at`: `asc`/`desc`|
| `search`     | string  | null    | Full-text search in conversation   |
| `date_from`  | string  | null    | ISO 8601 date filter (e.g., `2025-01-01`) |
| `date_to`    | string  | null    | ISO 8601 date filter              |

**Response `200 OK`:**

```json
{
  "conversations": [
    {
      "id": "cnv_a1b2c3d4",
      "title": "CRISPR-Cas9 off-target effects",
      "created_at": "2025-01-15T10:30:00Z",
      "updated_at": "2025-01-20T14:22:00Z",
      "message_count": 47,
      "document_count": 3,
      "tags": ["crispr", "gene-editing"],
      "model_used": "claude-3-5-sonnet",
      "last_message": {
        "id": "msg_xyz789",
        "role": "assistant",
        "content_preview": "The off-target cleavage activity of eSpCas9 variants...",
        "created_at": "2025-01-20T14:22:00Z"
      }
    }
  ],
  "pagination": {
    "has_more": true,
    "next_cursor": "eyJpZCI6ImNudl9hYmMxMjMifQ",
    "total_count": 156
  }
}
```

---

### POST /api/v1/chat/conversations

Create a new conversation.

**Request:**

```json
{
  "title": "CRISPR-Cas9 off-target effects",
  "tags": ["crispr", "gene-editing"],
  "model": "claude-3-5-sonnet",
  "system_prompt": "You are a helpful research assistant specializing in molecular biology.",
  "temperature": 0.7,
  "metadata": {
    "project_id": "proj_abc123",
    "grant_number": "NIH-R01-12345"
  }
}
```

| Field         | Type    | Required | Description                        |
|---------------|---------|----------|-----------------------------------|
| `title`       | string  | Yes      | Conversation title (max 200 chars)|
| `tags`        | string[]| No       | Array of tag strings              |
| `model`       | string  | No       | Model ID (defaults to user setting)|
| `system_prompt`| string | No       | Custom system instructions         |
| `temperature` | number  | No       | 0.0–1.0, default 0.7             |
| `metadata`    | object  | No       | Arbitrary key-value pairs         |

**Response `201 Created`:**

```json
{
  "id": "cnv_a1b2c3d4",
  "title": "CRISPR-Cas9 off-target effects",
  "created_at": "2025-01-20T14:22:00Z",
  "updated_at": "2025-01-20T14:22:00Z",
  "tags": ["crispr", "gene-editing"],
  "model_used": "claude-3-5-sonnet",
  "temperature": 0.7,
  "metadata": {
    "project_id": "proj_abc123",
    "grant_number": "NIH-R01-12345"
  }
}
```

---

### GET /api/v1/chat/conversations/{id}

Get a single conversation by ID.

**Response `200 OK`:**

```json
{
  "id": "cnv_a1b2c3d4",
  "title": "CRISPR-Cas9 off-target effects",
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-20T14:22:00Z",
  "message_count": 47,
  "document_count": 3,
  "tags": ["crispr", "gene-editing"],
  "model_used": "claude-3-5-sonnet",
  "temperature": 0.7,
  "metadata": {
    "project_id": "proj_abc123",
    "grant_number": "NIH-R01-12345"
  }
}
```

---

### PATCH /api/v1/chat/conversations/{id}

Update a conversation's title, tags, or metadata.

**Request:**

```json
{
  "title": "CRISPR-Cas9 Off-Target Analysis (Revised)",
  "tags": ["crispr", "gene-editing", "review"],
  "metadata": {
    "project_id": "proj_abc123",
    "status": "in-review"
  }
}
```

**Response `200 OK`:**

```json
{
  "id": "cnv_a1b2c3d4",
  "title": "CRISPR-Cas9 Off-Target Analysis (Revised)",
  "updated_at": "2025-01-20T15:00:00Z",
  "tags": ["crispr", "gene-editing", "review"],
  "metadata": {
    "project_id": "proj_abc123",
    "status": "in-review"
  }
}
```

---

### DELETE /api/v1/chat/conversations/{id}

Permanently delete a conversation and all its messages.

**Response `204 No Content`**

> **Warning:** This action is irreversible. All messages, documents, and memories associated with this conversation will be deleted.

---

### GET /api/v1/chat/conversations/{id}/messages

List messages in a conversation (for history display).

**Query Parameters:**

| Parameter    | Type    | Default | Description                        |
|--------------|---------|---------|-----------------------------------|
| `limit`      | integer | 50      | Items per page (max 200)          |
| `cursor`     | string  | null    | Pagination cursor                  |
| `include_metadata` | boolean | false | Include confidence/reasoning if available |

**Response `200 OK`:**

```json
{
  "messages": [
    {
      "id": "msg_aaa111",
      "role": "user",
      "content": "What are the main off-target effects of CRISPR-Cas9?",
      "created_at": "2025-01-20T14:22:00Z",
      "attachments": []
    },
    {
      "id": "msg_aaa222",
      "role": "assistant",
      "content": "CRISPR-Cas9 can cause off-target effects where...",
      "created_at": "2025-01-20T14:22:05Z",
      "sources": [
        {
          "document_id": "doc_xyz789",
          "chunk_id": "chk_abc123",
          "relevance_score": 0.94,
          "text_excerpt": "...off-target cleavage sites were identified..."
        }
      ]
    }
  ],
  "pagination": {
    "has_more": false,
    "total_count": 47
  }
}
```

---

### POST /api/v1/chat/conversations/{id}/message [SSE]

Send a message and receive a streaming response via Server-Sent Events (SSE).

#### Request

**Headers:**

```http
Content-Type: application/json
Accept: text/event-stream
```

**Body:**

```json
{
  "content": "Compare the off-target profiles of eSpCas9 and HiFi Cas9",
  "attachments": ["doc_xyz789", "doc_abc123"],
  "stream": true
}
```

| Field         | Type     | Required | Description                              |
|---------------|----------|----------|------------------------------------------|
| `content`     | string   | Yes      | Message content (max 10,000 chars)      |
| `attachments` | string[] | No       | Document IDs to include as context       |
| `stream`      | boolean  | No       | Enable streaming (default: true)         |

#### Enhanced Query Parameters (v2.0)

| Parameter              | Type    | Description                                        |
|------------------------|---------|---------------------------------------------------|
| `?include_confidence=true` | boolean | Add per-claim confidence scores to response   |
| `?include_reasoning=true` | boolean | Add chain-of-thought reasoning trace          |
| `?temporal=true`      | boolean | Enable temporal reasoning over time-ordered context |
| `?multi_hop=true`      | boolean | Enable multi-hop reasoning across documents      |

**Example with enhanced parameters:**

```http
POST /api/v1/chat/conversations/cnv_a1b2c3d4/message?include_confidence=true&include_reasoning=true&multi_hop=true
```

#### Response (SSE Events)

The response is a stream of SSE events. Each event type is described below:

##### `message_start`

```json
event: message_start
data: {"message_id": "msg_new_123", "conversation_id": "cnv_a1b2c3d4"}
```

##### `content_block_start`

```json
event: content_block_start
data: {"index": 0, "type": "text"}
```

##### `content_block_delta`

```json
event: content_block_delta
data: {"index": 0, "type": "text", "delta": "CRISPR-Cas9 off-target effects arise from..."}
```

##### `content_block_stop`

```json
event: content_block_stop
data: {"index": 0}
```

##### `sources` (with relevance data)

```json
event: sources
data: {
  "sources": [
    {
      "document_id": "doc_xyz789",
      "chunk_id": "chk_abc123",
      "title": "CRISPR Off-Target Analysis Paper",
      "relevance_score": 0.94,
      "text_excerpt": "...off-target cleavage sites in human genome...",
      "page_number": 3,
      "url": null
    },
    {
      "document_id": "doc_def456",
      "chunk_id": "chk_ghi789",
      "title": "eSpCas9 Optimization Study",
      "relevance_score": 0.89,
      "text_excerpt": "...high-fidelity variants reduce off-target...",
      "page_number": 12,
      "url": null
    }
  ]
}
```

##### `confidence` (when `include_confidence=true`)

```json
event: confidence
data: {
  "overall": 0.87,
  "claims": [
    {
      "claim": "eSpCas9 has 10-fold lower off-target activity",
      "confidence": 0.92,
      "supporting_sources": ["doc_xyz789"]
    },
    {
      "claim": "HiFi Cas9 uses modified sgRNA architecture",
      "confidence": 0.78,
      "supporting_sources": ["doc_def456"],
      "uncertainty_note": "Source does not provide direct comparison"
    }
  ]
}
```

##### `reasoning` (when `include_reasoning=true`)

```json
event: reasoning
data: {
  "steps": [
    {
      "step": 1,
      "thought": "The user is asking about off-target profiles comparison",
      "action": "Retrieving documents mentioning eSpCas9 or HiFi Cas9",
      "sources_consulted": ["doc_xyz789", "doc_def456"]
    },
    {
      "step": 2,
      "thought": "Found relevant sections in both documents about specificity",
      "action": "Comparing quantitative off-target rates",
      "documents_analyzed": 2,
      "relevant_chunks": 5
    },
    {
      "step": 3,
      "thought": "Synthesizing comparison based on retrieved evidence",
      "action": "Generating comparative response with citations",
      "claims_generated": 4
    }
  ]
}
```

##### `citation`

```json
event: citation
data: {
  "index": 156,
  "length": 24,
  "source": {
    "document_id": "doc_xyz789",
    "chunk_id": "chk_abc123",
    "relevance_score": 0.94
  }
}
```

##### `message_stop`

```json
event: message_stop
data: {"stop_reason": "end_turn"}
```

---

### Document Endpoints

#### POST /api/v1/chat/conversations/{id}/documents

Upload a document to a conversation for context.

**Request:** `multipart/form-data`

| Field      | Type   | Required | Description                          |
|------------|--------|----------|--------------------------------------|
| `file`     | file   | Yes      | PDF, DOCX, TXT, MD (max 50MB)        |
| `title`    | string | No       | Display title (defaults to filename)  |
| `metadata` | object | No       | Custom key-value metadata            |

**Supported file types:**

- `application/pdf`
- `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
- `text/plain`
- `text/markdown`

**Response `202 Accepted`:**

```json
{
  "document_id": "doc_xyz789",
  "filename": "crispr-paper.pdf",
  "title": "CRISPR Off-Target Analysis",
  "status": "processing",
  "created_at": "2025-01-20T14:22:00Z",
  "processing_progress": {
    "stage": "chunking",
    "percent_complete": 15
  }
}
```

---

#### GET /api/v1/chat/conversations/{id}/documents/{doc_id}/status

Check document processing status.

**Response `200 OK`:**

```json
{
  "document_id": "doc_xyz789",
  "status": "completed",
  "processing_info": {
    "stage": "completed",
    "percent_complete": 100,
    "chunks_created": 127,
    "entities_extracted": 43,
    "processing_time_ms": 4521
  },
  "metadata": {
    "page_count": 24,
    "word_count": 8432,
    "language": "en"
  }
}
```

Possible status values: `pending`, `processing`, `completed`, `failed`, `quarantined`

---

#### GET /api/v1/chat/conversations/{id}/documents

List all documents in a conversation.

**Response `200 OK`:**

```json
{
  "documents": [
    {
      "document_id": "doc_xyz789",
      "filename": "crispr-paper.pdf",
      "title": "CRISPR Off-Target Analysis",
      "status": "completed",
      "created_at": "2025-01-20T14:22:00Z",
      "page_count": 24,
      "chunks_indexed": 127
    }
  ],
  "pagination": {
    "has_more": false,
    "total_count": 3
  }
}
```

---

#### DELETE /api/v1/chat/conversations/{id}/documents/{doc_id}

Remove a document from a conversation and delete its indexed chunks.

**Response `204 No Content`**

---

## 4. Memory (Second Brain)

Orivory's memory system stores and retrieves knowledge using semantic search and temporal reasoning.

### POST /api/v1/memories

Create a new memory entry.

**Request:**

```json
{
  "content": "Dr. Chen's lab discovered that eSpCas9 shows 8-fold higher specificity compared to wild-type Cas9 in human cell lines.",
  "type": "finding",
  "tags": ["crispr", "espCas9", "specificity", "chen-lab"],
  "source_document_id": "doc_xyz789",
  "source_chunk_id": "chk_abc123",
  "confidence": 0.95,
  "expires_at": "2026-01-20T00:00:00Z",
  "metadata": {
    "experiment_id": "exp_001",
    "cell_line": "HEK293T"
  }
}
```

| Field              | Type     | Required | Description                            |
|--------------------|----------|----------|----------------------------------------|
| `content`          | string   | Yes      | Memory content (max 5,000 chars)      |
| `type`             | string   | No       | `finding`, `hypothesis`, `note`, `reference` (default: `note`) |
| `tags`             | string[] | No       | Array of tag strings                   |
| `source_document_id` | string | No       | Linked document ID                     |
| `source_chunk_id`  | string   | No       | Linked chunk ID                        |
| `confidence`       | number   | No       | 0.0–1.0 confidence score             |
| `expires_at`       | string   | No       | ISO 8601 expiration datetime           |
| `metadata`         | object   | No       | Arbitrary key-value pairs              |

**Response `201 Created`:**

```json
{
  "id": "mem_abc123def456",
  "content": "Dr. Chen's lab discovered that eSpCas9 shows 8-fold higher specificity...",
  "type": "finding",
  "tags": ["crispr", "espCas9", "specificity", "chen-lab"],
  "created_at": "2025-01-20T14:22:00Z",
  "updated_at": "2025-01-20T14:22:00Z",
  "source_document_id": "doc_xyz789",
  "source_chunk_id": "chk_abc123",
  "confidence": 0.95,
  "expires_at": "2026-01-20T00:00:00Z",
  "metadata": {
    "experiment_id": "exp_001",
    "cell_line": "HEK293T"
  }
}
```

---

### GET /api/v1/memories

List memories with filtering and semantic search.

**Query Parameters:**

| Parameter   | Type    | Default | Description                               |
|-------------|---------|---------|------------------------------------------|
| `q`         | string  | null    | Semantic search query                    |
| `type`      | string  | null    | Filter by type                           |
| `tags`      | string  | null    | Comma-separated tag filter               |
| `date_from` | string  | null    | ISO 8601 date filter                     |
| `date_to`   | string  | null    | ISO 8601 date filter                     |
| `limit`     | integer | 20      | Items per page (max 100)                 |
| `cursor`    | string  | null    | Pagination cursor                        |
| `sort`      | string  | `desc`  | Sort by `created_at`: `asc` or `desc`   |

**Response `200 OK`:**

```json
{
  "memories": [
    {
      "id": "mem_abc123def456",
      "content": "Dr. Chen's lab discovered that eSpCas9 shows 8-fold higher specificity...",
      "type": "finding",
      "tags": ["crispr", "espCas9", "specificity"],
      "created_at": "2025-01-20T14:22:00Z",
      "relevance_score": 0.94,
      "source_document_id": "doc_xyz789"
    }
  ],
  "pagination": {
    "has_more": false,
    "total_count": 24
  }
}
```

> **Note — `total` counts visible rows only:** dirty (stale-derived) memories are excluded from the page and from the total; superseded rows stay listed, labeled `state: "superseded"`.

> **Note — memory filter language (P1b, Qdrant):** the `where` object accepted by the memory search/recall path takes one operator per field, restricted to the allowlist `source_type`, `captured_at`, `salience`, `pinned`, `tags` (`user_id` is always the authenticated principal and is rejected as a filter). Values must be scalars (`bool`/`int`/`str`) or, for `$in`/`$nin`, a list. Tightened against the Chroma-era behaviour, each of these now raises `ValueError`: a **float** operand on `$eq`/`$ne`/`$in`/`$nin`/`$contains` (a float is a range question — use `$gt`/`$gte`/`$lt`/`$lte` on `salience`), a non-scalar operand where a scalar is required (e.g. `{"tags": {"$contains": ["a", "b"]}}` — `$contains` takes ONE element and matches it against the list), and range operators on fields that are not ranges (`pinned`, `tags`, `source_type`). `$ne` and `$nin` compile to `must_not` clauses, and because a missing field never matches an include, a memory carrying no `tags` key is still returned by `tags: {"$ne": "x"}`.

---

### GET /api/v1/memories/{id}

Get a single memory by ID.

**Response `200 OK`:**

```json
{
  "id": "mem_abc123def456",
  "content": "Dr. Chen's lab discovered that eSpCas9 shows 8-fold higher specificity...",
  "type": "finding",
  "tags": ["crispr", "espCas9", "specificity", "chen-lab"],
  "created_at": "2025-01-20T14:22:00Z",
  "updated_at": "2025-01-20T14:22:00Z",
  "source_document_id": "doc_xyz789",
  "source_chunk_id": "chk_abc123",
  "confidence": 0.95,
  "metadata": {
    "experiment_id": "exp_001",
    "cell_line": "HEK293T"
  }
}
```

---

### PATCH /api/v1/memories/{id}

Update a memory entry.

**Request:**

```json
{
  "content": "Updated content with additional findings...",
  "tags": ["crispr", "espCas9", "specificity", "chen-lab", "updated"],
  "confidence": 0.97
}
```

**Response `200 OK`:** Updated memory object.

---

### DELETE /api/v1/memories/{id}

Delete a memory entry.

**Response `204 No Content`**

---

### GET /api/v1/memories/digest

Generate a daily digest of recent memories organized by topic.

**Query Parameters:**

| Parameter    | Type    | Default   | Description                        |
|--------------|---------|-----------|-----------------------------------|
| `date`       | string  | today     | ISO 8601 date (e.g., `2025-01-20`) |
| `topic`      | string  | null      | Filter by primary topic tag        |
| `format`     | string  | `summary` | `summary`, `timeline`, `by-topic` |

**Response `200 OK`:**

```json
{
  "date": "2025-01-20",
  "topic": "crispr",
  "format": "by-topic",
  "digest": {
    "crispr": {
      "findings": 3,
      "summary": "Today's CRISPR research focused on off-target effects, with new data from Dr. Chen's lab confirming improved specificity of eSpCas9 variants.",
      "memories": [
        {
          "id": "mem_abc123",
          "content": "eSpCas9 shows 8-fold higher specificity...",
          "created_at": "2025-01-20T14:22:00Z"
        }
      ]
    },
    "gene-editing": {
      "findings": 1,
      "summary": "One new reference added regarding base editing applications."
    }
  },
  "generated_at": "2025-01-20T23:00:00Z"
}
```

---

### POST /api/v1/memories/recall

Semantic recall — search across all memories using natural language.

#### Standard Recall

**Request:**

```json
{
  "query": "What did we learn about eSpCas9 specificity?",
  "limit": 10,
  "tags": ["crispr"]
}
```

**Response `200 OK`:**

```json
{
  "query": "What did we learn about eSpCas9 specificity?",
  "results": [
    {
      "id": "mem_abc123def456",
      "content": "Dr. Chen's lab discovered that eSpCas9 shows 8-fold higher specificity...",
      "type": "finding",
      "relevance_score": 0.96,
      "tags": ["crispr", "espCas9", "specificity"],
      "source_document_id": "doc_xyz789",
      "created_at": "2025-01-20T14:22:00Z",
      "recalled_at": "2025-01-20T16:00:00Z"
    }
  ],
  "total_found": 3,
  "recalled_at": "2025-01-20T16:00:00Z"
}
```

#### Enhanced Recall with Temporal Reasoning (v2.0)

When `temporal=true`, the system orders results by temporal relevance and annotates temporal relationships between memories.

**Request:**

```json
{
  "query": "How has our understanding of Cas9 specificity evolved?",
  "temporal": true,
  "limit": 15
}
```

**Response:**

```json
{
  "query": "How has our understanding of Cas9 specificity evolved?",
  "results": [
    {
      "id": "mem_001",
      "content": "Initial paper review suggests Cas9 has significant off-target activity",
      "created_at": "2025-01-10T09:00:00Z",
      "temporal_position": "earliest",
      "temporal_annotation": {
        "position": "beginning",
        "relationship": "baseline"
      }
    },
    {
      "id": "mem_002",
      "content": "eSpCas9 variant shows reduced off-target cleavage",
      "created_at": "2025-01-15T11:30:00Z",
      "temporal_position": "intermediate",
      "temporal_annotation": {
        "position": "middle",
        "relationship": "confirmation",
        "builds_on": "mem_001"
      }
    },
    {
      "id": "mem_003",
      "content": "Chen lab confirms 8-fold specificity improvement with eSpCas9",
      "created_at": "2025-01-20T14:22:00Z",
      "temporal_position": "latest",
      "temporal_annotation": {
        "position": "current",
        "relationship": "refinement",
        "builds_on": "mem_002",
        "contradicts": null
      }
    }
  ],
  "temporal_insights": {
    "trend": "improving",
    "confidence_direction": "increasing",
    "span": "10 days",
    "trajectory_summary": "Understanding evolved from initial concern about off-target effects to confirmed improvement with engineered variants"
  },
  "total_found": 3
}
```

#### Enhanced Recall with Multi-Hop Reasoning (v2.0)

When `multi_hop=true`, the system chains across multiple memories to answer complex questions that require connecting disparate pieces of knowledge.

**Request:**

```json
{
  "query": "Is there a connection between Dr. Chen's eSpCas9 findings and the base editing research?",
  "multi_hop": true,
  "max_hops": 3,
  "limit": 20
}
```

**Response:**

```json
{
  "query": "Is there a connection between Dr. Chen's eSpCas9 findings and the base editing research?",
  "multi_hop": true,
  "results": [
    {
      "type": "direct",
      "id": "mem_chen_001",
      "content": "Dr. Chen's lab discovered that eSpCas9 shows 8-fold higher specificity...",
      "relevance_score": 0.94
    },
    {
      "type": "direct",
      "id": "mem_base_001",
      "content": "Base editing provides an alternative to double-strand breaks...",
      "relevance_score": 0.88
    }
  ],
  "chains": [
    {
      "chain_id": "chain_001",
      "length": 3,
      "path": ["mem_chen_001", "mem_common_001", "mem_base_001"],
      "connection_explanation": "Both Chen's eSpCas9 work and base editing research address the specificity challenge in genome editing. They share a common theme: reducing unintended genomic modifications.",
      "strength": 0.82,
      "intermediate_memory": {
        "id": "mem_common_001",
        "content": "The overarching goal across CRISPR variants is minimizing off-target genomic modifications"
      }
    }
  ],
  "insights": [
    {
      "type": "synthesis",
      "description": "eSpCas9 and base editing represent complementary approaches to the same fundamental challenge",
      "confidence": 0.79
    }
  ],
  "total_chains_found": 1,
  "total_found": 8
}
```

---

## 5. Knowledge Graph

### Entities

#### POST /api/v1/entities

Create a new entity.

**Request:**

```json
{
  "name": "eSpCas9",
  "type": "protein",
  "aliases": ["enhanced SpCas9", "high-fidelity Cas9"],
  "description": "Engineered Cas9 variant with reduced off-target activity",
  "properties": {
    "organism": "Streptococcus pyogenes",
    "modification_type": "point mutations",
    "specificity_improvement": "8-fold"
  },
  "source_document_id": "doc_xyz789",
  "source_chunk_id": "chk_abc123"
}
```

| Field               | Type     | Required | Description                            |
|--------------------|----------|----------|----------------------------------------|
| `name`             | string   | Yes      | Entity name (max 200 chars)            |
| `type`             | string   | Yes      | Entity type (see below)               |
| `aliases`          | string[] | No       | Alternative names                      |
| `description`      | string   | No       | Natural language description           |
| `properties`       | object   | No       | Structured key-value properties       |
| `source_document_id` | string | No       | Source document                        |
| `source_chunk_id`  | string   | No       | Source chunk                           |

**Entity Types:**

`protein`, `gene`, `disease`, `drug`, `cell_line`, `lab`, `researcher`, `paper`, `grant`, `concept`, `method`, `organism`, `compound`, `custom`

**Response `201 Created`:**

```json
{
  "id": "ent_abc123",
  "name": "eSpCas9",
  "type": "protein",
  "aliases": ["enhanced SpCas9", "high-fidelity Cas9"],
  "description": "Engineered Cas9 variant with reduced off-target activity",
  "properties": {
    "organism": "Streptococcus pyogenes",
    "modification_type": "point mutations",
    "specificity_improvement": "8-fold"
  },
  "created_at": "2025-01-20T14:22:00Z",
  "updated_at": "2025-01-20T14:22:00Z",
  "relation_count": 5,
  "source_document_id": "doc_xyz789"
}
```

---

#### GET /api/v1/entities

List all entities.

**Query Parameters:**

| Parameter   | Type    | Default | Description                        |
|-------------|---------|---------|-----------------------------------|
| `type`      | string  | null    | Filter by entity type             |
| `q`         | string  | null    | Search by name or description     |
| `limit`     | integer | 20      | Items per page (max 100)          |
| `cursor`    | string  | null    | Pagination cursor                 |

**Response `200 OK`:**

```json
{
  "entities": [
    {
      "id": "ent_abc123",
      "name": "eSpCas9",
      "type": "protein",
      "aliases": ["enhanced SpCas9"],
      "description": "Engineered Cas9 variant with reduced off-target activity",
      "relation_count": 5,
      "created_at": "2025-01-20T14:22:00Z"
    }
  ],
  "pagination": {
    "has_more": true,
    "next_cursor": "eyJpZCI6ImVudF9hYmMxMjMifQ",
    "total_count": 342
  }
}
```

---

#### GET /api/v1/entities/{id}

Get a single entity.

**Response `200 OK`:**

```json
{
  "id": "ent_abc123",
  "name": "eSpCas9",
  "type": "protein",
  "aliases": ["enhanced SpCas9", "high-fidelity Cas9"],
  "description": "Engineered Cas9 variant with reduced off-target activity",
  "properties": {
    "organism": "Streptococcus pyogenes",
    "modification_type": "point mutations",
    "specificity_improvement": "8-fold"
  },
  "created_at": "2025-01-20T14:22:00Z",
  "updated_at": "2025-01-20T14:22:00Z",
  "source_document_id": "doc_xyz789",
  "related_memories": [
    {
      "id": "mem_xyz789",
      "content": "Chen lab's eSpCas9 validation...",
      "created_at": "2025-01-20T14:22:00Z"
    }
  ]
}
```

---

#### PATCH /api/v1/entities/{id}

Update an entity.

**Request:**

```json
{
  "aliases": ["enhanced SpCas9", "high-fidelity Cas9", "eSpCas9 v1.1"],
  "properties": {
    "organism": "Streptococcus pyogenes",
    "specificity_improvement": "10-fold"
  }
}
```

**Response `200 OK`:** Updated entity object.

---

#### DELETE /api/v1/entities/{id}

Delete an entity and all its relations.

**Response `204 No Content`**

---

### Relations

#### POST /api/v1/relations

Create a relation between two entities.

**Request:**

```json
{
  "from_entity_id": "ent_abc123",
  "to_entity_id": "ent_def456",
  "relation_type": "improves",
  "strength": 0.92,
  "bidirectional": false,
  "description": "eSpCas9 shows improved specificity compared to wild-type Cas9",
  "source_document_id": "doc_xyz789"
}
```

| Field              | Type    | Required | Description                            |
|-------------------|---------|----------|----------------------------------------|
| `from_entity_id`  | string  | Yes      | Source entity ID                       |
| `to_entity_id`    | string  | Yes      | Target entity ID                       |
| `relation_type`   | string  | Yes      | Relation type (see below)             |
| `strength`        | number  | No       | 0.0–1.0, default 1.0                 |
| `bidirectional`   | boolean | No       | If true, creates reverse relation too  |
| `description`     | string  | No       | Natural language description           |
| `source_document_id` | string | No    | Source document                        |

**Relation Types:**

`interacts_with`, `regulates`, `inhibits`, `activates`, `associates_with`, `encodes`, `expressed_in`, `located_in`, `similar_to`, `improves`, `derived_from`, `parent_of`, `child_of`, `part_of`, `related_to`, `cites`, `authored_by`, `funded_by`, `contradicts`, `supports`, `custom`

**Response `201 Created`:**

```json
{
  "id": "rel_abc123",
  "from_entity_id": "ent_abc123",
  "to_entity_id": "ent_def456",
  "relation_type": "improves",
  "strength": 0.92,
  "bidirectional": false,
  "description": "eSpCas9 shows improved specificity compared to wild-type Cas9",
  "created_at": "2025-01-20T14:22:00Z"
}
```

---

#### GET /api/v1/relations

List relations with filtering.

**Query Parameters:**

| Parameter      | Type    | Default | Description                        |
|----------------|---------|---------|-----------------------------------|
| `from_entity_id` | string | null   | Filter by source entity           |
| `to_entity_id`   | string | null   | Filter by target entity           |
| `relation_type`  | string | null   | Filter by relation type           |
| `limit`          | integer | 20    | Items per page (max 100)          |
| `cursor`         | string  | null   | Pagination cursor                 |

**Response `200 OK`:**

```json
{
  "relations": [
    {
      "id": "rel_abc123",
      "from_entity_id": "ent_abc123",
      "to_entity_id": "ent_def456",
      "relation_type": "improves",
      "strength": 0.92,
      "description": "eSpCas9 shows improved specificity...",
      "created_at": "2025-01-20T14:22:00Z"
    }
  ],
  "pagination": {
    "has_more": false,
    "total_count": 127
  }
}
```

---

#### DELETE /api/v1/relations/{id}

Delete a relation.

**Response `204 No Content`**

---

### Graph Visualization Endpoints

#### GET /api/v1/graph/snapshot

Get the entire knowledge graph as nodes and edges.

**Query Parameters:**

| Parameter  | Type    | Default | Description                        |
|------------|---------|---------|-----------------------------------|
| `depth`    | integer | 2       | Traversal depth (max 5)           |
| `node_types` | string | null   | Comma-separated entity types      |
| `limit`    | integer | 500     | Max nodes to return (max 1000)    |

**Response `200 OK`:**

```json
{
  "nodes": [
    {
      "id": "ent_abc123",
      "name": "eSpCas9",
      "type": "protein",
      "properties": {
        "specificity_improvement": "8-fold"
      }
    },
    {
      "id": "ent_def456",
      "name": "wild-type Cas9",
      "type": "protein",
      "properties": {}
    }
  ],
  "edges": [
    {
      "id": "rel_abc123",
      "source": "ent_abc123",
      "target": "ent_def456",
      "type": "improves",
      "strength": 0.92
    }
  ],
  "stats": {
    "total_nodes": 2,
    "total_edges": 1,
    "node_type_distribution": {
      "protein": 2
    },
    "relation_type_distribution": {
      "improves": 1
    }
  },
  "generated_at": "2025-01-20T16:00:00Z"
}
```

---

#### GET /api/v1/graph/clusters

Identify topic clusters in the knowledge graph using community detection.

**Query Parameters:**

| Parameter   | Type    | Default | Description                        |
|-------------|---------|---------|-----------------------------------|
| `algorithm` | string  | `louvain` | Algorithm: `louvain`, `label_propagation` |
| `threshold` | number  | 0.3    | Minimum relation strength         |

**Response `200 OK`:**

```json
{
  "clusters": [
    {
      "cluster_id": "cluster_001",
      "name": "CRISPR Specificity",
      "entities": [
        {
          "id": "ent_abc123",
          "name": "eSpCas9",
          "type": "protein"
        },
        {
          "id": "ent_def456",
          "name": "wild-type Cas9",
          "type": "protein"
        }
      ],
      "relations": 5,
      "density": 0.83,
      "primary_topic": "genome editing specificity"
    }
  ],
  "algorithm": "louvain",
  "total_clusters": 7,
  "generated_at": "2025-01-20T16:00:00Z"
}
```

---

#### GET /api/v1/graph/related/{entity_name}

Find entities and relations connected to a specific entity.

**Path Parameters:**

| Parameter      | Type   | Description                              |
|---------------|--------|----------------------------------------|
| `entity_name` | string | Entity name (URL-encoded)              |

**Query Parameters:**

| Parameter  | Type    | Default | Description                        |
|------------|---------|---------|-----------------------------------|
| `max_depth` | integer | 2      | Maximum traversal depth (max 4)   |
| `relation_types` | string | null | Comma-separated relation types  |
| `limit`    | integer | 50      | Max related entities (max 200)   |

**Example:**

```http
GET /api/v1/graph/related/eSpCas9?max_depth=2&limit=20
```

**Response `200 OK`:**

```json
{
  "center_entity": {
    "id": "ent_abc123",
    "name": "eSpCas9",
    "type": "protein"
  },
  "related": [
    {
      "entity": {
        "id": "ent_def456",
        "name": "wild-type Cas9",
        "type": "protein"
      },
      "relation": {
        "type": "improves",
        "strength": 0.92,
        "direction": "outgoing"
      },
      "depth": 1
    },
    {
      "entity": {
        "id": "ent_chenlab",
        "name": "Chen Lab",
        "type": "lab"
      },
      "relation": {
        "type": "authored_by",
        "strength": 1.0,
        "direction": "outgoing"
      },
      "depth": 2,
      "path": ["ent_abc123", "ent_paper001", "ent_chenlab"]
    }
  ],
  "stats": {
    "total_related": 15,
    "unique_depths": [1, 2],
    "entity_types": ["protein", "lab", "paper", "method"]
  }
}
```

---

## 6. Sources & Connectors

### Sources

#### POST /api/v1/sources

Register an external source (RSS feed, URL, or connector).

**Request (RSS):**

```json
{
  "type": "rss",
  "name": "Nature Biotechnology RSS",
  "url": "https://www.nature.com/nbt/rss",
  "schedule": "daily",
  "tags": ["nature", "biotechnology", "crispr"],
  "filters": {
    "keywords_include": ["crispr", "gene editing"],
    "keywords_exclude": ["clinical trial phase 3"]
  }
}
```

**Request (URL / Web Clipper):**

```json
{
  "type": "webpage",
  "name": "eSpCas9 Paper",
  "url": "https://www.science.org/doi/10.1126/science.aa",
  "tags": ["espCas9", "specificity"]
}
```

**Request (Connector):**

```json
{
  "type": "connector",
  "connector_type": "zotero",
  "name": "Zotero Library",
  "config": {
    "library_id": "zotero-lib-123"
  },
  "schedule": "weekly",
  "tags": ["zotero", "library"]
}
```

| Field            | Type    | Required | Description                            |
|-----------------|---------|----------|----------------------------------------|
| `type`          | string  | Yes      | `rss`, `webpage`, `connector`          |
| `name`          | string  | Yes      | Display name                           |
| `url`           | string  | Conditional | Required for `rss` and `webpage`    |
| `schedule`      | string  | No       | `hourly`, `daily`, `weekly`, `manual` |
| `tags`          | string[]| No       | Tag strings                            |
| `filters`       | object  | No       | Keyword filters                        |
| `connector_type`| string  | Conditional | Required for `connector` type       |
| `config`        | object  | Conditional | Required for `connector` type       |

**Connector Types:** `zotero`, `endnote`, `mendeley`, `pocket`, `instapaper`, `readwise`

**Response `201 Created`:**

```json
{
  "id": "src_abc123",
  "type": "rss",
  "name": "Nature Biotechnology RSS",
  "status": "active",
  "last_sync": null,
  "next_sync": "2025-01-21T00:00:00Z",
  "items_captured": 0,
  "created_at": "2025-01-20T14:22:00Z"
}
```

---

#### GET /api/v1/sources

List all registered sources.

**Response `200 OK`:**

```json
{
  "sources": [
    {
      "id": "src_abc123",
      "type": "rss",
      "name": "Nature Biotechnology RSS",
      "status": "active",
      "last_sync": "2025-01-20T00:00:00Z",
      "next_sync": "2025-01-21T00:00:00Z",
      "items_captured": 47,
      "schedule": "daily"
    }
  ],
  "pagination": {
    "has_more": false,
    "total_count": 5
  }
}
```

---

#### DELETE /api/v1/sources/{id}

Remove a source and optionally delete all associated documents.

**Query Parameters:**

| Parameter              | Type    | Default | Description                        |
|------------------------|---------|---------|-----------------------------------|
| `delete_documents`     | boolean | false   | Also delete all indexed documents |

**Response `204 No Content`**

---

#### POST /api/v1/sources/{id}/sync

Manually trigger a sync for a source.

**Response `202 Accepted`:**

```json
{
  "source_id": "src_abc123",
  "status": "queued",
  "estimated_items": 12,
  "queue_position": 1
}
```

---

### Sync Endpoints

#### GET /api/v1/sources/{id}/sync/status

Check sync status for a source.

**Response `200 OK`:**

```json
{
  "source_id": "src_abc123",
  "sync_status": "completed",
  "started_at": "2025-01-20T14:22:00Z",
  "completed_at": "2025-01-20T14:22:45Z",
  "items_processed": 12,
  "items_new": 3,
  "items_updated": 9,
  "errors": []
}
```

---

## 7. Feedback & Calibration (NEW)

### POST /api/v1/feedback

Submit feedback on an answer to improve model calibration.

**Request:**

```json
{
  "query_id": "msg_new_123",
  "answer_id": "msg_aaa222",
  "feedback_type": "accuracy",
  "rating": 4,
  "details": {
    "was_helpful": true,
    "was_accurate": true,
    "was_complete": false,
    "had_hallucination": false,
    "had_omission": true,
    "omission_description": "Did not mention the 2024 update to HiFi Cas9 specificity data",
    "citation_correct": true
  },
  "corrections": [
    {
      "claim": "HiFi Cas9 uses modified sgRNA architecture",
      "correction": "This is incorrect. HiFi Cas9 uses high-fidelity Cas9 protein mutations, not modified sgRNA."
    }
  ],
  "preferred_response": "The correct description should include the point mutations in the HNH domain..."
}
```

| Field               | Type     | Required | Description                            |
|--------------------|----------|----------|----------------------------------------|
| `query_id`         | string   | Yes      | ID of the user query message          |
| `answer_id`       | string   | Yes      | ID of the assistant answer             |
| `feedback_type`   | string   | Yes      | `accuracy`, `relevance`, `format`, `safety` |
| `rating`          | integer  | Yes      | 1–5 star rating                       |
| `details`         | object   | No       | Structured feedback details           |
| `corrections`     | object[] | No       | Array of factual corrections           |
| `preferred_response` | string | No       | User's preferred answer                |

**Feedback Types:**

- `accuracy`: Factual correctness of the answer
- `relevance`: Whether the answer addressed the query
- `format`: Presentation quality (citations, structure, etc.)
- `safety`: Content safety concerns

**Response `201 Created`:**

```json
{
  "feedback_id": "fb_abc123",
  "query_id": "msg_new_123",
  "answer_id": "msg_aaa222",
  "feedback_type": "accuracy",
  "rating": 4,
  "submitted_at": "2025-01-20T14:22:00Z",
  "calibration_impact": {
    "model_confidence_adjustment": -0.03,
    "calibration_updated": true,
    "affected_claims": ["claim_001", "claim_003"]
  }
}
```

---

### GET /api/v1/feedback/accuracy

Get accuracy metrics for the authenticated user's feedback history.

**Query Parameters:**

| Parameter   | Type    | Default | Description                        |
|-------------|---------|---------|-----------------------------------|
| `period`    | string  | `30d`   | `7d`, `30d`, `90d`, `all`        |
| `model`     | string  | null    | Filter by model ID                |

**Response `200 OK`:**

```json
{
  "period": "30d",
  "metrics": {
    "total_feedback": 47,
    "average_rating": 4.2,
    "rating_distribution": {
      "1": 2,
      "2": 1,
      "3": 5,
      "4": 24,
      "5": 15
    },
    "accuracy_score": 0.89,
    "relevance_score": 0.92,
    "format_score": 0.85,
    "improvement_trend": "increasing",
    "trend_delta": "+0.04"
  },
  "feedback_by_type": {
    "accuracy": {
      "count": 28,
      "avg_rating": 4.1,
      "corrections_submitted": 3
    },
    "relevance": {
      "count": 12,
      "avg_rating": 4.5
    },
    "format": {
      "count": 7,
      "avg_rating": 3.9
    }
  },
  "calibration_status": {
    "calibrated": true,
    "last_calibration": "2025-01-19T10:00:00Z",
    "calibration_interval": "weekly",
    "next_scheduled": "2025-01-26T10:00:00Z"
  },
  "model_breakdown": {
    "claude-3-5-sonnet": {
      "accuracy_score": 0.91,
      "feedback_count": 32
    },
    "gpt-4o": {
      "accuracy_score": 0.86,
      "feedback_count": 15
    }
  },
  "generated_at": "2025-01-20T16:00:00Z"
}
```

---

### POST /api/v1/feedback/calibration

Trigger a recalibration of the model's confidence estimates based on accumulated feedback.

**Request:**

```json
{
  "scope": "targeted",
  "target_model": "claude-3-5-sonnet",
  "focus_areas": ["protein_entities", "crispr_methods"],
  "force": false
}
```

| Field         | Type    | Required | Description                            |
|---------------|---------|----------|----------------------------------------|
| `scope`       | string  | No       | `full`, `targeted`, `incremental` (default: `incremental`) |
| `target_model`| string  | No       | Specific model to recalibrate          |
| `focus_areas` | string[]| No       | Entity/relation types to focus on      |
| `force`       | boolean | No       | Force recalibration even if not due    |

**Response `202 Accepted`:**

```json
{
  "calibration_id": "cal_abc123",
  "status": "queued",
  "scope": "targeted",
  "target_model": "claude-3-5-sonnet",
  "focus_areas": ["protein_entities", "crispr_methods"],
  "estimated_completion": "2025-01-20T16:05:00Z",
  "feedback_samples_used": 142,
  "threshold_met": true
}
```

---

## 8. Insights (NEW)

### GET /api/v1/insights/unexpected

**"What I Didn't Know I Knew"** — Discover unexpected connections and knowledge gaps filled by your corpus.

**Query Parameters:**

| Parameter   | Type    | Default | Description                              |
|-------------|---------|---------|----------------------------------------|
| `limit`     | integer | 10      | Number of insights (max 50)             |
| `threshold` | number  | 0.6     | Minimum surprise score (0.0–1.0)        |
| `categories` | string | null    | Comma-separated: `connection`, `gap`, `pattern` |

**Response `200 OK`:**

```json
{
  "insights": [
    {
      "id": "ins_abc123",
      "category": "connection",
      "type": "unexpected_bridge",
      "title": "Unexpected connection between CRISPR base editing and RNA research",
      "description": "Analysis revealed that your base editing research corpus contains 3 papers that cite RNA helicase mechanisms, connecting to your prior RNA research.",
      "surprise_score": 0.84,
      "entities_involved": [
        {"id": "ent_be", "name": "Base Editing", "type": "method"},
        {"id": "ent_rna", "name": "RNA Helicases", "type": "protein"}
      ],
      "supporting_evidence": [
        {
          "document_id": "doc_paper1",
          "title": "RNA-guided base editing with helicase co-factors",
          "relevance": 0.91
        }
      ],
      "potential_use_cases": [
        "Cross-validate base editing efficiency using RNA helicase assays",
        "Explore helicase-assisted delivery mechanisms"
      ],
      "discovered_at": "2025-01-20T16:00:00Z"
    },
    {
      "id": "ins_def456",
      "category": "gap",
      "type": "knowledge_gap",
      "title": "Knowledge gap: Recent developments in prime editing",
      "description": "Your corpus contains limited coverage of prime editing (2 papers) compared to base editing (18 papers), despite prime editing representing a significant advancement.",
      "surprise_score": 0.72,
      "gap_details": {
        "topic": "prime editing",
        "existing_coverage": 2,
        "recommended_coverage": 15,
        "gap_ratio": 0.13
      },
      "suggested_sources": [
        "Anzalone et al. (2019) — Prime editing original paper",
        "Chen et al. (2021) — Prime editing 2.0"
      ],
      "discovered_at": "2025-01-20T16:00:00Z"
    },
    {
      "id": "ins_ghi789",
      "category": "pattern",
      "type": "temporal_pattern",
      "title": "Increasing focus on specificity in recent research",
      "description": "Over the past 6 months, 67% of new papers added focus on specificity optimization, compared to 34% in the prior period.",
      "surprise_score": 0.65,
      "pattern_details": {
        "metric": "specificity_mentions",
        "current_period_ratio": 0.67,
        "prior_period_ratio": 0.34,
        "change_direction": "increasing",
        "change_magnitude": "+0.33"
      },
      "discovered_at": "2025-01-20T16:00:00Z"
    }
  ],
  "summary": {
    "total_insights": 10,
    "by_category": {
      "connection": 4,
      "gap": 3,
      "pattern": 3
    },
    "avg_surprise_score": 0.71
  },
  "generated_at": "2025-01-20T16:00:00Z"
}
```

---

### GET /api/v1/insights/connections

Discover multi-hop connections across your knowledge graph.

**Query Parameters:**

| Parameter     | Type    | Default | Description                              |
|--------------|---------|---------|----------------------------------------|
| `entity_a`   | string  | null    | Start entity name                       |
| `entity_b`   | string  | null    | Target entity name                      |
| `max_hops`   | integer | 3       | Maximum path length (max 5)            |
| `relation_types` | string | null  | Comma-separated relation types to allow |
| `limit`      | integer | 10      | Number of connection sets (max 50)     |

**Response `200 OK`:**

```json
{
  "query": {
    "entity_a": "eSpCas9",
    "entity_b": "RNA helicase",
    "max_hops": 3
  },
  "connections": [
    {
      "connection_id": "conn_001",
      "path": [
        {
          "entity": {"id": "ent_esp", "name": "eSpCas9", "type": "protein"},
          "relation": null,
          "depth": 0
        },
        {
          "entity": {"id": "ent_delivery", "name": "AAV Delivery", "type": "method"},
          "relation": {"type": "uses", "strength": 0.88},
          "depth": 1
        },
        {
          "entity": {"id": "ent_rna", "name": "RNA Helicase", "type": "protein"},
          "relation": {"type": "associated_with", "strength": 0.72},
          "depth": 2
        }
      ],
      "path_length": 2,
      "path_strength": 0.63,
      "explanation": "eSpCas9 research often discusses AAV delivery methods, which involve RNA helicase activity for unpackaging.",
      "confidence": 0.78,
      "supporting_sources": [
        {
          "document_id": "doc_delivery_1",
          "title": "AAV Vector Unpackaging in Neurons",
          "relevance": 0.85
        }
      ]
    },
    {
      "connection_id": "conn_002",
      "path": [
        {
          "entity": {"id": "ent_esp", "name": "eSpCas9", "type": "protein"},
          "relation": null,
          "depth": 0
        },
        {
          "entity": {"id": "ent_paper1", "name": "High-Fidelity Cas9 Review", "type": "paper"},
          "relation": {"type": "reviewed_in", "strength": 1.0},
          "depth": 1
        },
        {
          "entity": {"id": "ent_rna", "name": "RNA Helicase", "type": "protein"},
          "relation": {"type": "cited_by", "strength": 0.65},
          "depth": 2
        }
      ],
      "path_length": 2,
      "path_strength": 0.65,
      "explanation": "The high-fidelity Cas9 review paper cites a reference discussing RNA helicase interactions.",
      "confidence": 0.71,
      "supporting_sources": []
    }
  ],
  "statistics": {
    "total_connections": 2,
    "avg_path_length": 2.0,
    "avg_path_strength": 0.64,
    "entity_type_pairs": [
      {"from_type": "protein", "to_type": "method", "count": 1},
      {"from_type": "protein", "to_type": "paper", "count": 1}
    ]
  },
  "generated_at": "2025-01-20T16:00:00Z"
}
```

---

## 9. Admin API

> **Note:** Admin endpoints require an admin-scoped access token. Contact support to obtain admin credentials.

### User Management

#### GET /api/v1/admin/users

List all users (admin only).

**Query Parameters:**

| Parameter  | Type    | Default | Description                        |
|------------|---------|---------|-----------------------------------|
| `limit`    | integer | 50      | Items per page (max 100)          |
| `cursor`   | string  | null    | Pagination cursor                 |
| `plan`     | string  | null    | Filter by plan: `free`, `pro`, `enterprise` |
| `status`   | string  | null    | Filter by status: `active`, `suspended`, `pending` |

**Response `200 OK`:**

```json
{
  "users": [
    {
      "id": "usr_a1b2c3d4",
      "email": "researcher@university.edu",
      "full_name": "Dr. Jane Smith",
      "plan": "pro",
      "status": "active",
      "created_at": "2025-01-15T10:30:00Z",
      "last_login": "2025-01-20T14:22:00Z",
      "usage": {
        "conversations_created": 47,
        "documents_indexed": 12,
        "memories_created": 243,
        "api_calls_this_month": 3847
      }
    }
  ],
  "pagination": {
    "has_more": true,
    "next_cursor": "eyJ1c2VyX2lkIjoiMTIzNDU2Nzg5YWJjZGVmIn0",
    "total_count": 1523
  }
}
```

---

#### PATCH /api/v1/admin/users/{user_id}

Update a user's account status, plan, or quotas.

**Request:**

```json
{
  "plan": "enterprise",
  "status": "active",
  "quotas": {
    "conversations_per_day": 500,
    "documents_per_month": 1000,
    "api_calls_per_month": 100000
  }
}
```

**Response `200 OK`:**

```json
{
  "id": "usr_a1b2c3d4",
  "plan": "enterprise",
  "status": "active",
  "quotas": {
    "conversations_per_day": 500,
    "documents_per_month": 1000,
    "api_calls_per_month": 100000
  },
  "updated_at": "2025-01-20T16:00:00Z"
}
```

---

### Diagnostics

#### GET /api/v1/admin/diagnostics

Get system health and performance diagnostics.

**Query Parameters:**

| Parameter   | Type    | Default | Description                        |
|-------------|---------|---------|-----------------------------------|
| `component` | string  | null    | Filter by component               |

**Components:** `api`, `vector_store`, `llm_gateway`, `document_processor`, `memory_store`, `graph_engine`, `sync_service`

**Response `200 OK`:**

```json
{
  "timestamp": "2025-01-20T16:00:00Z",
  "overall_status": "healthy",
  "components": {
    "api": {
      "status": "healthy",
      "latency_p50_ms": 45,
      "latency_p95_ms": 120,
      "latency_p99_ms": 250,
      "error_rate": 0.002
    },
    "vector_store": {
      "status": "healthy",
      "latency_p50_ms": 12,
      "latency_p95_ms": 35,
      "error_rate": 0.0001
    },
    "llm_gateway": {
      "status": "healthy",
      "latency_p50_ms": 850,
      "latency_p95_ms": 2100,
      "error_rate": 0.008
    },
    "document_processor": {
      "status": "degraded",
      "queue_depth": 847,
      "processing_rate_per_min": 12,
      "error_rate": 0.015
    }
  }
}
```

---

### Quality Metrics

#### GET /api/v1/admin/metrics/quality

Get aggregated quality metrics across all users.

**Query Parameters:**

| Parameter  | Type    | Default | Description                        |
|------------|---------|---------|-----------------------------------|
| `period`   | string  | `30d`   | `7d`, `30d`, `90d`               |

**Response `200 OK`:**

```json
{
  "period": "30d",
  "metrics": {
    "total_queries": 487293,
    "total_users": 1523,
    "avg_answer_quality_score": 0.87,
    "avg_source_relevance_score": 0.84,
    "avg_citation_accuracy": 0.92,
    "hallucination_rate": 0.008,
    "user_satisfaction": {
      "nps_score": 67,
      "avg_rating": 4.3,
      "response_rate": 0.99
    },
    "model_performance": {
      "claude-3-5-sonnet": {
        "usage_share": 0.62,
        "avg_quality_score": 0.89,
        "avg_latency_ms": 1200
      },
      "gpt-4o": {
        "usage_share": 0.38,
        "avg_quality_score": 0.85,
        "avg_latency_ms": 1500
      }
    }
  },
  "generated_at": "2025-01-20T16:00:00Z"
}
```

---

## 10. Webhooks (Future)

Webhooks will allow your application to receive real-time notifications when events occur in Orivory.

### Event Types

| Event                    | Description                                      |
|-------------------------|------------------------------------------------|
| `conversation.created`  | A new conversation was created                  |
| `conversation.updated`  | A conversation was updated                     |
| `document.processed`    | A document finished processing                 |
| `document.failed`       | A document failed processing                   |
| `memory.created`        | A new memory was created                       |
| `source.synced`         | A source finished syncing                      |
| `user.quota_exceeded`   | A user exceeded their API quota                |

### Payload Format

```json
{
  "event": "document.processed",
  "timestamp": "2025-01-20T14:22:00Z",
  "data": {
    "document_id": "doc_xyz789",
    "conversation_id": "cnv_a1b2c3d4",
    "status": "completed",
    "chunks_created": 127
  },
  "signature": "sha256=..."
}
```

### Security

Each webhook delivery includes an `X-Orivory-Signature` header containing an HMAC-SHA256 signature of the payload, using your webhook secret.

**Verification Example (Python):**

```python
import hmac
import hashlib

def verify_webhook(payload_body: bytes, secret: str, signature_header: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload_body,
        hashlib.sha256
    ).hexdigest()
    expected_header = f"sha256={expected}"
    return hmac.compare_digest(expected_header, signature_header)
```

---

## 11. Rate Limits & Quotas

### Per-Tier Limits

| Feature                   | Free      | Pro        | Enterprise    |
|--------------------------|-----------|------------|--------------|
| API requests/minute       | 60        | 300        | 1,000        |
| API requests/day          | 1,000     | 10,000     | 100,000      |
| Conversations             | 50 total  | Unlimited  | Unlimited    |
| Documents indexed         | 5         | 500/month  | 10,000/month |
| Memory entries            | 200       | 10,000     | Unlimited    |
| Entities                  | 100       | 5,000      | Unlimited    |
| SSE streaming              | Yes       | Yes        | Yes          |
| RAG retrieval              | Yes       | Yes        | Yes          |
| Knowledge graph            | View only | Full access| Full access  |
| Temporal reasoning         | No        | Yes        | Yes          |
| Multi-hop reasoning        | No        | Yes        | Yes          |
| Confidence scoring         | No        | Yes        | Yes          |
| Chain-of-thought          | No        | Yes        | Yes          |
| Webhooks                   | No        | Yes        | Yes          |
| Admin API                  | No        | No         | Yes          |

### Rate Limit Headers

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 284
X-RateLimit-Reset: 1704067460
X-RateLimit-Policy: 300;w=60
```

### Quota Exceeded Response

When a quota is exceeded, the API returns `429 Too Many Requests`:

```json
{
  "error": {
    "code": "QUOTA_EXCEEDED",
    "message": "Daily document quota exceeded. Upgrade to Pro for 500 documents/month.",
    "details": {
      "quota_type": "documents_per_month",
      "current_usage": 500,
      "quota_limit": 500,
      "reset_date": "2025-02-01T00:00:00Z",
      "upgrade_url": "/pricing"
    },
    "request_id": "req_abc123"
  }
}
```

---

## 12. Error Reference

### Standard Error Format

Every error response follows this structure:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description.",
    "details": {},
    "request_id": "req_abc123xyz"
  }
}
```

### Error Code Index

| Code                       | HTTP Status | Description & Common Causes                               |
|---------------------------|-------------|----------------------------------------------------------|
| `VALIDATION_ERROR`        | 400         | Invalid request body or query parameters               |
| `UNAUTHORIZED`            | 401         | Missing or invalid access token                        |
| `TOKEN_EXPIRED`           | 401         | Access token has expired                               |
| `REFRESH_TOKEN_INVALID`   | 401         | Refresh token is invalid, revoked, or expired          |
| `FORBIDDEN`               | 403         | Insufficient permissions for this resource             |
| `RESOURCE_NOT_FOUND`      | 404         | Entity does not exist or is not accessible             |
| `CONVERSATION_NOT_FOUND`  | 404         | Conversation ID does not exist                         |
| `DOCUMENT_NOT_FOUND`      | 404         | Document ID does not exist                             |
| `MEMORY_NOT_FOUND`        | 404         | Memory ID does not exist                               |
| `ENTITY_NOT_FOUND`        | 404         | Entity does not exist in knowledge graph               |
| `CONFLICT`                | 409         | Resource with same identifier already exists            |
| `DUPLICATE_EMAIL`         | 409         | Email already registered                               |
| `UNPROCESSABLE_ENTITY`    | 422         | Semantically invalid input (e.g., circular relation)   |
| `FILE_TOO_LARGE`          | 413         | Document exceeds 50MB limit                          |
| `UNSUPPORTED_FILE_TYPE`   | 415         | File type not supported (see supported types)          |
| `RATE_LIMIT_EXCEEDED`     | 429         | Too many requests; see `Retry-After` header           |
| `QUOTA_EXCEEDED`          | 429         | Monthly or daily quota exceeded                        |
| `SERVICE_UNAVAILABLE`     | 503         | Maintenance window or system overload                  |
| `INTERNAL_ERROR`          | 500         | Unexpected server error; include `request_id` in bug reports |

### Troubleshooting

| Symptom                       | Likely Cause                    | Solution                               |
|------------------------------|---------------------------------|---------------------------------------|
| `401 UNAUTHORIZED` on all requests | Token not passed or expired  | Re-authenticate via `/auth/login`      |
| `429 RATE_LIMIT_EXCEEDED`   | Too many rapid requests         | Implement exponential backoff          |
| `429 QUOTA_EXCEEDED`        | Monthly limit reached            | Wait for reset or upgrade plan         |
| `503 SERVICE_UNAVAILABLE`   | System maintenance              | Check status page, retry after 5 min   |
| SSE stream stalls            | Connection timeout              | Reconnect with fresh token            |
| Document processing stuck    | Large file or encoding issue     | Split file or convert to PDF/TXT      |

---

## 13. Agent Clients & MCP Hub

The Open Memory Hub lets external AI agents (Claude Desktop, Claude Code, OpenClaw, Cursor, custom agents) read and write your memory over MCP (Model Context Protocol). Access is controlled by **agent clients**: per-agent tokens with scoped permissions (`memory:read` / `memory:write`), and every authorized call is recorded in an **access ledger** — which AI read or wrote what, and when.

> **Note:** The MCP endpoint is mounted at `/mcp` and is gated by the `MCP_HUB_ENABLED` setting (`true` by default; set `MCP_HUB_ENABLED=false` in `.env` to disable it).

### POST /api/v1/agents

Register an agent client. Returns the plaintext token — **shown exactly once**. Only a SHA-256 hash is stored; the token cannot be recovered or re-displayed afterwards.

**Request:**

```bash
curl -s -X POST https://api.orivory.io/api/v1/agents \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Claude Desktop", "scopes": ["memory:read", "memory:write"]}'
```

| Field    | Type     | Required | Description                                                       |
|----------|----------|----------|-------------------------------------------------------------------|
| `name`   | string   | Yes      | Display name (max 100 chars)                                      |
| `scopes` | string[] | No       | `memory:read` (search/get/list), `memory:write` (add/delete). Defaults to `["memory:read"]` |

**Response `201 Created`:**

```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "name": "Claude Desktop",
  "scopes": ["memory:read", "memory:write"],
  "status": "active",
  "created_at": "2026-09-02T10:30:00Z",
  "token": "oa_9f8e7d6c5b4a3210fedcba9876543210"
}
```

> **Warning: the token is shown once.** Copy it immediately and store it in your MCP client's config — it will never appear in any API response again. Anyone who loses it must revoke the client and register a new one.

### GET /api/v1/agents

List the current user's agent clients, newest first. Never includes token material.

**Response `200 OK`:**

```json
{
  "items": [
    {
      "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "name": "Claude Desktop",
      "scopes": ["memory:read", "memory:write"],
      "status": "active",
      "created_at": "2026-09-02T10:30:00Z",
      "last_used_at": "2026-09-02T14:22:00Z",
      "revoked_at": null
    }
  ],
  "total": 1
}
```

### DELETE /api/v1/agents/{id}

Revoke an agent client. The token stops working immediately and revocation is idempotent (re-revoking the caller's own already-revoked client still returns `204`).

**Response `204 No Content`**

### GET /api/v1/agents/access-log

The access ledger: every authorized MCP call, newest first. Which agent did what, when.

**Query Parameters:**

| Parameter         | Type    | Default | Description                              |
|-------------------|---------|---------|------------------------------------------|
| `agent_client_id` | string  | null    | Filter by agent client ID                |
| `limit`           | integer | 50      | Items per page (max 200)                 |
| `offset`          | integer | 0       | Offset for pagination                    |

**Response `200 OK`:**

```json
{
  "items": [
    {
      "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
      "agent_client_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "action": "mcp_search",
      "memory_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "detail": {"query": "CRISPR specificity"},
      "created_at": "2026-09-02T14:22:00Z"
    }
  ],
  "total": 1
}
```

Ledger `action` values: `mcp_search`, `mcp_get`, `mcp_list`, `mcp_add`, `mcp_delete`, `mcp_forget`. `memory_id` is null for search/list actions.

---

### MCP Endpoint

The hub speaks **streamable HTTP MCP** at:

```
http://localhost:8000/mcp
```

Authentication uses your agent token in either header:

```http
Authorization: Bearer oa_9f8e7d6c5b4a3210fedcba9876543210
```

```http
X-Orivory-Agent-Token: oa_9f8e7d6c5b4a3210fedcba9876543210
```

Requests without a valid token — or with a revoked token — are rejected before any tool runs; only authorized calls are ledgered.

#### Tools

| Tool            | Scope         | Description                                          |
|-----------------|---------------|------------------------------------------------------|
| `search_memory` | `memory:read` | Semantic search over the caller's memory hub          |
| `get_memory`    | `memory:read` | Fetch one memory by ID                                |
| `list_recent`   | `memory:read` | List the caller's most recent memories                |
| `add_memory`    | `memory:write`| Store a new memory (title, content, optional tags)    |
| `delete_memory` | `memory:write`| Delete one memory by ID                               |
| `forget_memory` | `memory:write`| Erase memories with cascades + verification receipt (see [§14](#14-erasure-receipts)) |

Scopes are enforced per call: a token with only `memory:read` cannot `add_memory` or `delete_memory`.

#### Connecting an MCP Client (Claude Desktop example)

```json
{
  "mcpServers": {
    "orivory-memory": {
      "type": "http",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "X-Orivory-Agent-Token": "oa_9f8e7d6c5b4a3210fedcba9876543210"
      }
    }
  }
}
```

> **Note — DNS-rebinding protection is on by default (localhost-only).** `/mcp` is constructed without explicit transport security, and in `mcp` 1.29.x FastMCP defaults `host="127.0.0.1"`, which auto-enables localhost-only DNS-rebind protection — so **any non-localhost `Host` header is answered `421` as shipped** (connection attempts against a public hostname will fail until configured).
>
> To accept other hostnames — e.g. behind a reverse proxy that forwards a public `Host` — set `MCP_HUB_ALLOWED_HOSTS=your.host.example` in the environment (comma-separated for several). This explicitly enables `TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=[...])` on the FastMCP instance.
>
> Until configured, keep the endpoint localhost-bound or proxy with `Host` preservation pointing at localhost.

---

## 14. Erasure Receipts

Erasing a memory removes the row **and every derived artifact** (child memories, entity links, source links, vector-store entries), then runs a post-deletion verification pass: re-query the vector store and re-count residual DB rows per target. Each erasure call returns one **receipt** with per-target detail. Receipts are user-scoped and are deleted with the user.

> **Honest v0 verification:** v0 verifies erasure by **absence-checks** — the receipt confirms that vectors and DB rows are *gone*. It does not probe whether facts can be re-inferred from correlated knowledge-graph data (KG-correlation re-inference probing is a planned follow-up). Also note that `Entity`/`Relation` nodes themselves survive memory erasure in v0 (link counts are recorded in the receipt; orphan pruning is a follow-up). Don't market this as "adversarially verified" until the deeper protocol ships.

### POST /api/v1/erasure-receipts

Erase memories owned by the caller. Foreign or unknown ids are recorded in the receipt as `"not_found_or_foreign"` — never deleted, no existence leak. Erasure is best-effort per target: one failing target never aborts the other targets.

**Request:**

```bash
curl -s -X POST https://api.orivory.io/api/v1/erasure-receipts \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"memory_ids": ["3fa85f64-5717-4562-b3fc-2c963f66afa6"]}'
```

| Field        | Type   | Required | Description                                    |
|--------------|--------|----------|------------------------------------------------|
| `memory_ids` | UUID[] | Yes      | 1–100 memory ids (duplicates are deduplicated) |

**Response `201 Created`:** one receipt. `detail.targets[]` carries one entry per requested id — `affected_memory_ids` (transitive descendants erased with the target), `traversal_depth`, `entity_links` / `source_links` counts, `vectors_deleted`, `vector_residual`, `vector_residual_checked`, and `db_residual`:

```json
{
  "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "user_id": "5f0b3a2e-1c4d-4e8f-9a7b-2d6c8e1f0a3b",
  "requested_memory_ids": ["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
  "status": "completed",
  "detail": {
    "requested_by": "rest_api",
    "targets": [
      {
        "memory_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "status": "deleted",
        "affected_memory_ids": [],
        "traversal_depth": 0,
        "entity_links": 2,
        "source_links": 1,
        "vectors_deleted": ["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        "vector_residual": [],
        "vector_residual_checked": true,
        "db_residual": {"children": 0, "entity_links": 0, "source_links": 0}
      }
    ],
    "summary": {"requested": 1, "erased": 1, "skipped": 0, "errors": 0, "residual_vectors": 0, "residual_rows": 0}
  },
  "created_at": "2026-09-02T14:22:00Z"
}
```

**Receipt status semantics:**

| Status                    | Meaning                                                              |
|---------------------------|----------------------------------------------------------------------|
| `completed`               | Every target erased and positively verified — absence readback confirmed for every deleted target |
| `completed_unverified`    | Erasure succeeded and found no residual, but no positive presence readback (a target is `pending` or `unknown`); spec §5.4 / P1 gate: `completed` is never stored without one |
| `completed_with_residual` | Erasure succeeded, but the verification pass found leftover vectors or DB rows |
| `completed_with_errors`   | At least one target's erasure raised; the error is recorded per-target and the remaining targets were still erased |

Rollup precedence: `completed_with_errors` > `completed_with_residual` > `completed_unverified` > `completed`. `detail.verification`/`detail.index_pending` are omitted when the call erased nothing (a forget that deleted nothing verified nothing).

`vector_residual_checked: false` means the vector-store re-query was unavailable during verification (the DB delete still succeeded — Postgres is the source of truth). A `false` flag alone does not imply residual data.

**Reconciliation.** Open receipts (`completed_unverified`) are re-checked by the reconcile pass (`POST /admin/erasure/reconcile`), OLDEST first (`created_at ASC`, ties broken by id; at most 200 per pass — FIFO, so a sustained erase load cannot starve an old receipt out of the window). A pass that reads the index clean rewrites the SAME receipt to `completed` with the re-verified evidence. A pass that still finds residual vectors does NOT relabel the receipt: it records what it observed in `detail` (`vector_residual_checked: true`, `vector_residual: [...]`) and leaves `status` alone, so the receipt stays open and re-checkable by a later pass — never frozen into the terminal `completed_with_residual`. A pass whose readback failed observed nothing and writes nothing at all: the receipt is byte-identical afterwards. The pass is upgrade-only: `completed`, `completed_with_residual` and `completed_with_errors` receipts are not even scanned, and no open receipt is ever downgraded.

### GET /api/v1/erasure-receipts

List the current user's receipts, newest first.

**Query Parameters:**

| Parameter | Type    | Default | Description                              |
|-----------|---------|---------|------------------------------------------|
| `limit`   | integer | 50      | Items per page (max 200)                 |
| `offset`  | integer | 0       | Offset for pagination                    |

**Response `200 OK`:**

```json
{
  "items": [
    {
      "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
      "user_id": "5f0b3a2e-1c4d-4e8f-9a7b-2d6c8e1f0a3b",
      "requested_memory_ids": ["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
      "status": "completed",
      "detail": {"targets": [], "summary": {}},
      "created_at": "2026-09-02T14:22:00Z"
    }
  ],
  "total": 1
}
```

```bash
curl -s "https://api.orivory.io/api/v1/erasure-receipts?limit=50&offset=0" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

### GET /api/v1/erasure-receipts/{id}

Fetch one receipt. Unknown or other users' receipts return `404` (no existence leak).

```bash
curl -s https://api.orivory.io/api/v1/erasure-receipts/7c9e6679-7425-40de-944b-e07fc1f90ae7 \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

### MCP: forget_memory

Agents with the `memory:write` scope can call the `forget_memory` MCP tool (endpoint `/mcp`, see [§13](#13-agent-clients--mcp-hub)) with `{"memory_ids": ["<uuid>", ...]}`. It returns a compact summary (`receipt_id`, `status`, `erased`, `skipped`, `invalid`) instead of the full receipt — fetch the receipt via `GET /api/v1/erasure-receipts/{id}` for the per-target detail. Every authorized call appends an `mcp_forget` row to the access ledger pointing at the receipt.

> **Note — ledger rows survive erasure by design.** The access ledger is append-only and is never erased by an erasure call: it records that a memory was *accessed before* deletion, which is exactly what makes it an audit trail. Receipts, by contrast, quote personal memory ids and are deleted with the user.

---

## 15. Import Paths

Bring an existing AI assistant's memory into Orivory in one call. The endpoint accepts a raw export file, auto-detects (or takes an explicit format), normalizes it into memories, and returns a summary.

### POST /api/v1/imports

```bash
curl -X POST https://api.orivory.io/api/v1/imports \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "file=@conversations.json" \
  -F "source_format=chatgpt"
```

| Form field | Required | Description |
|---|---|---|
| `file` | yes | The raw export file (JSON). Max **20 MiB** (larger → `413`). |
| `source_format` | no | `auto` (same as omitted/blank — detection) · `chatgpt` · `claude` · `generic`. An explicit unknown value → `422`. |

**Accepted formats** (detection heuristics follow the [PAM importer mappings](https://github.com/portable-ai-memory/portable-ai-memory/blob/master/importer-mappings.md)):

| Format | Detected shape | Notes |
|---|---|---|
| `chatgpt` | OpenAI export: array of conversations with a `mapping` DAG of `{author.role, content.parts}` nodes | One memory per conversation; system/empty turns skipped; turns ordered by `create_time` |
| `claude` | Claude export: `chat_messages` with `sender` human/assistant and message-level `text` | Thinking/tool blocks dropped |
| `generic` | JSON array of `{title?, content, created_at?, url?, ref?, tags?}` | The escape hatch for anything else |
| PAM bundle | `{"schema": "portable-ai-memory", "memories": [...]}` | [Portable AI Memory](https://portable-ai-memory.org/spec/v1.0/) `memory-store.json` (auto-detects to `generic`) |

**Response `201`** — `ImportSummary`:

```json
{
  "parsed": 42,
  "created": 40,
  "skipped_duplicates": 2,
  "failed": 0,
  "index_failures": 0
}
```

Duplicates are detected per `(user, source_type, source_ref)` — re-uploading the same export skips what you already imported. Dedup runs per-request (select-then-insert): two concurrent uploads of the same file can both succeed, and generic items without a `ref` field are re-created on every re-upload (a unique index is the planned hardening). Conversation content is clipped at **10,000 characters** (truncation marker appended). Malformed entries inside an otherwise-valid file are **skipped, never fatal** — one bad conversation can't fail the whole import. Embedding is best-effort: `index_failures > 0` means those memories exist and are searchable by keyword but not yet vector-indexed (Postgres is the source of truth; reindex tasks recover them).

**Errors:** `422` — unparseable file, undetectable format, explicit unknown format, undecodable bytes · `413` — file over 20 MiB.

> **Honest notes:**
> - **Rewind/Limitless: no adapter.** Their local history lives in a SQLCipher-encrypted SQLite database with no official export format (the app was sunset 2025-12-19). Convert manually to the generic JSON shape, or wait for a dedicated adapter.
> - **ChatGPT "Memory" feature contents are NOT in the data export** — only conversations.
> - **OpenRecall** stores an unencrypted local SQLite; one `sqlite3` query converts it to the generic JSON shape:
>
>   ```bash
>   sqlite3 recall.db "SELECT json_group_array(json_object('content', text, 'created_at', timestamp, 'title', title)) FROM entries" > openrecall.json
>   ```

---

## Appendix A: OpenAPI Schema (YAML)

```yaml
openapi: 3.1.0
info:
  title: Orivory API
  version: '2.0'
  description: RAG-native answer engine for researchers

servers:
  - url: https://api.orivory.io/api/v1
    description: Production

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  schemas:
    Error:
      type: object
      properties:
        error:
          type: object
          properties:
            code:
              type: string
            message:
              type: string
            details:
              type: object
            request_id:
              type: string

    Pagination:
      type: object
      properties:
        has_more:
          type: boolean
        next_cursor:
          type: string
        total_count:
          type: integer

    Conversation:
      type: object
      properties:
        id:
          type: string
        title:
          type: string
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time
        message_count:
          type: integer
        document_count:
          type: integer
        tags:
          type: array
          items:
            type: string

    Memory:
      type: object
      properties:
        id:
          type: string
        content:
          type: string
        type:
          type: string
          enum: [finding, hypothesis, note, reference]
        tags:
          type: array
          items:
            type: string
        confidence:
          type: number
        created_at:
          type: string
          format: date-time

    Entity:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        type:
          type: string
        description:
          type: string
        properties:
          type: object
        relation_count:
          type: integer

paths:
  /auth/register:
    post:
      operationId: registerUser
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [email, password, full_name, accept_terms]
              properties:
                email:
                  type: string
                  format: email
                password:
                  type: string
                  minLength: 8
                full_name:
                  type: string
                research_focus:
                  type: string
                accept_terms:
                  type: boolean
      responses:
        '201':
          description: User registered successfully
        '400':
          description: Validation error

  /auth/login:
    post:
      operationId: loginUser
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [email, password]
              properties:
                email:
                  type: string
                password:
                  type: string
      responses:
        '200':
          description: Login successful
          content:
            application/json:
              schema:
                type: object
                properties:
                  access_token:
                    type: string
                  refresh_token:
                    type: string
                  user:
                    $ref: '#/components/schemas/User'
        '401':
          description: Invalid credentials

  /chat/conversations:
    get:
      operationId: listConversations
      security:
        - bearerAuth: []
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
        - name: cursor
          in: query
          schema:
            type: string
      responses:
        '200':
          description: List of conversations
    post:
      operationId: createConversation
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [title]
              properties:
                title:
                  type: string
                tags:
                  type: array
                  items:
                    type: string
                model:
                  type: string
      responses:
        '201':
          description: Conversation created

  /chat/conversations/{id}/message:
    post:
      operationId: sendMessage
      security:
        - bearerAuth: []
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
        - name: include_confidence
          in: query
          schema:
            type: boolean
          description: Include per-claim confidence scores
        - name: include_reasoning
          in: query
          schema:
            type: boolean
          description: Include chain-of-thought reasoning
        - name: temporal
          in: query
          schema:
            type: boolean
          description: Enable temporal reasoning
        - name: multi_hop
          in: query
          schema:
            type: boolean
          description: Enable multi-hop reasoning
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [content]
              properties:
                content:
                  type: string
                attachments:
                  type: array
                  items:
                    type: string
      responses:
        '200':
          description: Streaming response via SSE
          content:
            text/event-stream:
              schema:
                type: string

  /memories:
    get:
      operationId: listMemories
      security:
        - bearerAuth: []
      parameters:
        - name: q
          in: query
          schema:
            type: string
          description: Semantic search query
        - name: type
          in: query
          schema:
            type: string
        - name: limit
          in: query
          schema:
            type: integer
    post:
      operationId: createMemory
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/Memory'
      responses:
        '201':
          description: Memory created

  /memories/recall:
    post:
      operationId: recallMemories
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                limit:
                  type: integer
                tags:
                  type: array
                  items:
                    type: string
                temporal:
                  type: boolean
                  description: Enable temporal reasoning
                multi_hop:
                  type: boolean
                  description: Enable multi-hop reasoning
                max_hops:
                  type: integer
                  default: 3
      responses:
        '200':
          description: Memory recall results

  /feedback:
    post:
      operationId: submitFeedback
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [query_id, answer_id, feedback_type, rating]
              properties:
                query_id:
                  type: string
                answer_id:
                  type: string
                feedback_type:
                  type: string
                  enum: [accuracy, relevance, format, safety]
                rating:
                  type: integer
                  minimum: 1
                  maximum: 5
                details:
                  type: object
                corrections:
                  type: array
                  items:
                    type: object
      responses:
        '201':
          description: Feedback submitted

  /feedback/accuracy:
    get:
      operationId: getAccuracyMetrics
      security:
        - bearerAuth: []
      parameters:
        - name: period
          in: query
          schema:
            type: string
            enum: [7d, 30d, 90d, all]
            default: 30d
      responses:
        '200':
          description: Accuracy metrics

  /feedback/calibration:
    post:
      operationId: triggerCalibration
      security:
        - bearerAuth: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                scope:
                  type: string
                  enum: [full, targeted, incremental]
                target_model:
                  type: string
                focus_areas:
                  type: array
                  items:
                    type: string
                force:
                  type: boolean
      responses:
        '202':
          description: Calibration triggered

  /insights/unexpected:
    get:
      operationId: getUnexpectedInsights
      security:
        - bearerAuth: []
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 10
        - name: threshold
          in: query
          schema:
            type: number
            default: 0.6
      responses:
        '200':
          description: Unexpected insights

  /insights/connections:
    get:
      operationId: getConnections
      security:
        - bearerAuth: []
      parameters:
        - name: entity_a
          in: query
          schema:
            type: string
        - name: entity_b
          in: query
          schema:
            type: string
        - name: max_hops
          in: query
          schema:
            type: integer
            default: 3
        - name: limit
          in: query
          schema:
            type: integer
            default: 10
      responses:
        '200':
          description: Multi-hop connections
```

---

## Appendix B: SDK Examples

### Python

```python
import orivory

client = orivory.Client(
    api_key="your_api_key",
    base_url="https://api.orivory.io/api/v1"
)

# Authenticate
auth = client.auth.login(
    email="researcher@university.edu",
    password="SecureP@ssw0rd!"
)
client.set_token(auth.access_token)

# Create a conversation
conversation = client.chat.conversations.create(
    title="CRISPR off-target analysis",
    tags=["crispr", "gene-editing"]
)

# Send a streaming message with enhanced parameters
with client.chat.conversations(conversation.id).message_stream(
    content="Compare eSpCas9 vs HiFi Cas9 specificity profiles",
    include_confidence=True,
    include_reasoning=True,
    multi_hop=True
) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            print(event.delta, end="", flush=True)
        elif event.type == "confidence":
            print(f"\n\n[Confidence: {event.overall:.0%}]")

# Create a memory
memory = client.memories.create(
    content="eSpCas9 shows 8-fold improved specificity over wild-type",
    type="finding",
    tags=["crispr", "espCas9", "specificity"]
)

# Recall with temporal reasoning
results = client.memories.recall(
    query="How has understanding of Cas9 specificity evolved?",
    temporal=True,
    limit=10
)
for result in results.results:
    print(f"{result.temporal_position}: {result.content}")

# Submit feedback
feedback = client.feedback.submit(
    query_id="msg_123",
    answer_id="msg_456",
    feedback_type="accuracy",
    rating=4,
    details={"was_accurate": True, "had_omission": True}
)

# Get unexpected insights
insights = client.insights.unexpected(threshold=0.7)
for insight in insights.insights:
    print(f"[{insight.category}] {insight.title}")
```

---

### JavaScript / TypeScript

```typescript
import { Orivory } from '@orivory/sdk';

const client = new Orivory({
  apiKey: process.env.Orivory_API_KEY,
  baseUrl: 'https://api.orivory.io/api/v1'
});

// Authenticate
const auth = await client.auth.login({
  email: 'researcher@university.edu',
  password: 'SecureP@ssw0rd!'
});
client.setToken(auth.accessToken);

// Create conversation
const conversation = await client.chat.conversations.create({
  title: 'CRISPR off-target analysis',
  tags: ['crispr', 'gene-editing'],
  model: 'claude-3-5-sonnet'
});

// Stream message with enhanced parameters
const stream = client.chat.conversations(conversation.id).messageStream({
  content: 'Compare eSpCas9 vs HiFi Cas9 specificity profiles',
  includeConfidence: true,
  includeReasoning: true,
  multiHop: true
});

for await (const event of stream) {
  switch (event.type) {
    case 'content_block_delta':
      process.stdout.write(event.delta);
      break;
    case 'confidence':
      console.log(`\n\n[Confidence: ${(event.overall * 100).toFixed(0)}%]`);
      break;
    case 'sources':
      console.log('\n\nSources:', event.sources);
      break;
    case 'reasoning':
      console.log('\n\nReasoning steps:', event.steps);
      break;
  }
}

// Create memory
const memory = await client.memories.create({
  content: 'eSpCas9 shows 8-fold improved specificity over wild-type',
  type: 'finding',
  tags: ['crispr', 'espCas9', 'specificity']
});

// Recall with temporal reasoning
const recallResults = await client.memories.recall({
  query: 'How has understanding of Cas9 specificity evolved?',
  temporal: true,
  limit: 10
});

// Recall with multi-hop reasoning
const hopResults = await client.memories.recall({
  query: 'Connection between eSpCas9 and base editing?',
  multiHop: true,
  maxHops: 3
});

// Submit feedback
await client.feedback.submit({
  queryId: 'msg_123',
  answerId: 'msg_456',
  feedbackType: 'accuracy',
  rating: 4,
  details: {
    wasAccurate: true,
    hadOmission: true,
    omissionDescription: 'Missing 2024 update to HiFi data'
  }
});

// Get unexpected insights
const insights = await client.insights.unexpected({
  threshold: 0.7,
  limit: 20
});

// Get multi-hop connections
const connections = await client.insights.connections({
  entityA: 'eSpCas9',
  entityB: 'RNA helicase',
  maxHops: 3
});
```

---

### cURL

```bash
# Authenticate
TOKEN=$(curl -s -X POST https://api.orivory.io/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"researcher@university.edu","password":"SecureP@ssw0rd!"}' \
  | jq -r '.access_token')

# Create conversation
CONV_ID=$(curl -s -X POST https://api.orivory.io/api/v1/chat/conversations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"CRISPR analysis","tags":["crispr"]}' \
  | jq -r '.id')

# Send streaming message with enhanced params
curl -X POST "https://api.orivory.io/api/v1/chat/conversations/$CONV_ID/message?include_confidence=true&include_reasoning=true&multi_hop=true" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"content":"Compare eSpCas9 vs HiFi Cas9 specificity","stream":true}'

# Create memory
curl -X POST https://api.orivory.io/api/v1/memories \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content":"eSpCas9 shows 8-fold specificity improvement","type":"finding","tags":["crispr","espCas9"]}'

# Recall with temporal reasoning
curl -X POST https://api.orivory.io/api/v1/memories/recall \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"How did our understanding evolve?","temporal":true}'

# Submit feedback
curl -X POST https://api.orivory.io/api/v1/feedback \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query_id":"msg_123","answer_id":"msg_456","feedback_type":"accuracy","rating":4}'

# Get accuracy metrics
curl -X GET "https://api.orivory.io/api/v1/feedback/accuracy?period=30d" \
  -H "Authorization: Bearer $TOKEN"

# Get unexpected insights
curl -X GET "https://api.orivory.io/api/v1/insights/unexpected?threshold=0.7" \
  -H "Authorization: Bearer $TOKEN"

# Get connections
curl -X GET "https://api.orivory.io/api/v1/insights/connections?entity_a=eSpCas9&entity_b=RNA+helicase&max_hops=3" \
  -H "Authorization: Bearer $TOKEN"
```

---

*Document version: 2.0 | Last updated: 2026-09-02*
