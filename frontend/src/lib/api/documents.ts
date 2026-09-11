/**
 * Documents API Client
 * Handles document upload, processing, and management
 */

import { apiClient } from "@/lib/api-client";

export interface Document {
  id: string;
  filename: string;
  title?: string;
  file_type: string;
  file_size: number;
  status: "processing" | "ready" | "error" | "partial";
  processing_progress?: number;
  error_message?: string;
  workspace_id?: string;
  tags?: string[];
  metadata?: {
    page_count?: number;
    word_count?: number;
    author?: string;
    created_date?: string;
    [key: string]: any;
  };
  summary?: string;
  created_at: Date;
  updated_at: Date;
  last_accessed_at?: Date;
}

export interface UploadedFile {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  status: Document["status"];
}

export interface UploadProgress {
  documentId: string;
  progress: number;
  status: "uploading" | "processing" | "ready" | "error";
  message?: string;
}

export interface UploadParams {
  file: File;
  workspace_id?: string;
  title?: string;
  tags?: string[];
  metadata?: Record<string, any>;
  onProgress?: (progress: UploadProgress) => void;
}

/**
 * List all documents from all conversations
 */
export async function listDocuments(): Promise<Document[]> {
  const token = typeof window !== "undefined" ? localStorage.getItem("auth_token") : null;
  if (!token) return [];

  try {
    // Get all conversations
    const sessionsRes = await fetch(`${apiClient.getBaseUrl()}/api/v1/chat/sessions`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const sessions = await sessionsRes.json();
    
    // Get documents for each conversation
    const allDocs: Document[] = [];
    for (const session of sessions) {
      try {
        const docsRes = await fetch(`${apiClient.getBaseUrl()}/api/v1/chat/conversations/${session.id}/documents`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (docsRes.ok) {
          const docs = await docsRes.json();
          allDocs.push(...docs);
        }
      } catch (e) {
        // Skip failed requests
      }
    }
    
    return allDocs.sort((a, b) => 
      new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );
  } catch (e) {
    console.error("Failed to list documents:", e);
    return [];
  }
}

/**
 * Upload a document with progress tracking
 * Uploads to the current chat session
 */
export async function uploadDocument({
  file,
  workspace_id,
  title,
  tags,
  metadata,
  onProgress,
}: UploadParams): Promise<Document> {
  // Get the current conversation/session ID from localStorage or use default
  let sessionId = typeof window !== "undefined" ? localStorage.getItem("current_session_id") : null;
  
  // If no session, create one first. POST /chat/sessions returns a SINGLE
  // SessionResponse object (not an array) — reading sessions[0] always missed
  // and spawned a stray empty conversation per upload.
  if (!sessionId) {
    const token = localStorage.getItem("auth_token");
    const response = await fetch(`${apiClient.getBaseUrl()}/api/v1/chat/sessions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
      body: JSON.stringify({}),
    });
    if (!response.ok) {
      throw new Error(`Failed to create chat session: ${response.status}`);
    }
    const session = await response.json();
    sessionId = session?.id;
    if (sessionId) {
      localStorage.setItem("current_session_id", sessionId);
    }
  }

  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress({
          documentId: "",
          progress: Math.round((e.loaded / e.total) * 100),
          status: "uploading",
          message: `Uploading: ${Math.round((e.loaded / e.total) * 100)}%`,
        });
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const data = JSON.parse(xhr.responseText);
        // Map backend response to frontend Document format
        const doc: Document = {
          id: data.id,
          filename: data.filename,
          title: data.filename,
          file_type: data.mime_type || "application/octet-stream",
          file_size: data.file_size,
          status: data.status === "pending" ? "processing" : (data.status as Document["status"]),
          created_at: new Date(data.created_at),
          updated_at: new Date(data.updated_at),
        };
        resolve(doc);
      } else {
        reject(new Error(`Upload failed: ${xhr.status} - ${xhr.responseText}`));
      }
    });

    xhr.addEventListener("error", () => {
      reject(new Error("Upload failed: Network error"));
    });

    // Use apiClient's baseUrl for consistency
    const baseUrl = apiClient.getBaseUrl();
    const endpoint = sessionId 
      ? `${baseUrl}/api/v1/chat/conversations/${sessionId}/documents`
      : `${baseUrl}/api/v1/chat/documents`;
      
    xhr.open("POST", endpoint);
    
    // Add auth header
    const token = typeof window !== "undefined" ? localStorage.getItem("auth_token") : null;
    if (token) {
      xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    }

    xhr.send(formData);
  });
}



/**
 * Delete a document
 * Uses the chat endpoint to delete documents
 */
export async function deleteDocument(id: string, conversationId?: string): Promise<void> {
  const token = typeof window !== "undefined" ? localStorage.getItem("auth_token") : null;
  if (!token) throw new Error("Not authenticated");

  // If we have the conversation ID, use the chat endpoint
  if (conversationId) {
    const response = await fetch(`${apiClient.getBaseUrl()}/api/v1/chat/conversations/${conversationId}/documents/${id}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) {
      throw new Error(`Failed to delete document: ${response.status}`);
    }
    return;
  }

  // Fallback: the backend serves DELETE /chat/documents/{id} (root delete,
  // no conversation needed). Failures throw — the previous version swallowed
  // them with console.warn, leaving the UI out of sync with the server.
  const response = await fetch(`${apiClient.getBaseUrl()}/api/v1/chat/documents/${id}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok && response.status !== 204) {
    throw new Error(`Failed to delete document: ${response.status}`);
  }
}

/**
 * Get supported file types
 */
/**
 * File types the backend actually ingests (app/services/document_service.py
 * ALLOWED_MIME + connector parsing). Keep in sync — the uploader validates
 * against this before hitting the API, and the dropzone badge list is
 * derived from these labels.
 */
export const SUPPORTED_FILE_TYPES = {
  pdf: { extensions: [".pdf"], icon: "📄", label: "PDF" },
  doc: { extensions: [".doc", ".docx"], icon: "📝", label: "Word Document" },
  text: { extensions: [".txt", ".md", ".rtf"], icon: "📃", label: "Text / Markdown" },
};

export const SUPPORTED_EXTENSIONS = Object.values(SUPPORTED_FILE_TYPES)
  .flatMap((t) => t.extensions)
  .join(",");

export const SUPPORTED_BADGES = Object.values(SUPPORTED_FILE_TYPES).map((t) => t.label);

export const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
export const MAX_FILES_PER_BATCH = 10;
