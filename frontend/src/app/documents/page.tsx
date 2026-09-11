"use client";

import { useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { DocumentUploader } from "@/components/documents/DocumentUploader";
import { listDocuments, uploadDocument, type Document } from "@/lib/api/documents";
import { 
  FileText, 
  Search, 
  Filter, 
  Upload, 
  Trash2, 
  MoreVertical,
  File,
  Image,
  FileSpreadsheet,
  Link
} from "lucide-react";

const FILE_ICONS: Record<string, any> = {
  pdf: FileText,
  doc: FileText,
  docx: FileText,
  txt: FileText,
  md: FileText,
  image: Image,
  jpg: Image,
  jpeg: Image,
  png: Image,
  gif: Image,
  webp: Image,
  xls: FileSpreadsheet,
  xlsx: FileSpreadsheet,
  csv: FileSpreadsheet,
  url: Link,
};

function FileIcon({ type, className }: { type: string; className?: string }) {
  const extension = type.toLowerCase().split("/").pop() || "";
  const Icon = FILE_ICONS[extension] || FILE_ICONS[type.split("/")[0]] || File;
  return <Icon className={className} />;
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState<"list" | "upload">("list");

  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true);
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (error) {
      console.error("Failed to fetch documents:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // Poll while any document is still processing so status chips settle
  // without a manual reload (ingestion runs async in Celery).
  useEffect(() => {
    const hasPending = documents.some((d) => d.status === "processing");
    if (!hasPending) return;
    const t = setInterval(() => {
      // Silent refresh: don't flash skeletons on each poll.
      listDocuments()
        .then(setDocuments)
        .catch(() => {});
    }, 5000);
    return () => clearInterval(t);
  }, [documents]);

  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to delete this document?")) return;
    
    try {
      const { deleteDocument } = await import("@/lib/api/documents");
      await deleteDocument(id);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
    } catch (error) {
      console.error("Failed to delete document:", error);
      alert("Failed to delete document. Please try again.");
    }
  };

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-6 py-8 max-w-6xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500/20 to-pink-500/20 border border-violet-500/30 flex items-center justify-center">
              <FileText className="w-6 h-6 text-violet-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Documents</h1>
              <p className="text-sm text-white/50">Manage your uploaded files</p>
            </div>
          </div>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab(activeTab === "list" ? "upload" : "list")}
            className={cn(
              "px-4 py-2 rounded-xl text-sm font-medium",
              "transition-all duration-300",
              activeTab === "upload"
                ? "bg-white/10 text-white/70"
                : "bg-gradient-to-r from-violet-600 to-purple-600 text-white shadow-lg shadow-violet-500/20"
            )}
          >
            {activeTab === "list" ? (
              <span className="flex items-center gap-2">
                <Upload className="w-4 h-4" />
                Upload
              </span>
            ) : (
              "View Documents"
            )}
          </motion.button>
        </div>

        {activeTab === "upload" ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-2xl mx-auto"
          >
            <DocumentUploader onUploadComplete={fetchDocuments} />
          </motion.div>
        ) : (
          <>
            {/* Search */}
            <div className="mb-6">
              <div className={cn(
                "relative rounded-xl border",
                "bg-white/[0.02] border-white/[0.08]",
                "focus-within:border-violet-500/40",
                "transition-all"
              )}>
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search documents..."
                  className={cn(
                    "w-full pl-11 pr-4 py-3",
                    "bg-transparent",
                    "text-white placeholder:text-white/30 text-sm",
                    "focus:outline-none"
                  )}
                />
              </div>
            </div>

            {/* Document list */}
            {loading ? (
              <div className="space-y-3">
                {[1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="h-20 rounded-xl bg-white/[0.02] border border-white/[0.05] animate-pulse"
                  />
                ))}
              </div>
            ) : documents.length === 0 ? (
              <div className="text-center py-16">
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-white/[0.03] border border-white/[0.08] flex items-center justify-center">
                  <FileText className="w-8 h-8 text-white/20" />
                </div>
                <p className="text-sm font-medium text-white/60 mb-1">No documents yet</p>
                <p className="text-xs text-white/30 mb-4">Upload your first document to get started</p>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setActiveTab("upload")}
                  className="px-4 py-2 rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 text-white text-sm font-medium shadow-lg shadow-violet-500/20"
                >
                  Upload Documents
                </motion.button>
              </div>
            ) : (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-2"
              >
                {documents.map((doc) => (
                  <motion.div
                    key={doc.id}
                    whileHover={{ x: 4 }}
                    className={cn(
                      "flex items-center gap-4 p-4 rounded-xl",
                      "bg-white/[0.02] border border-white/[0.08]",
                      "hover:border-white/[0.15] hover:bg-white/[0.04]",
                      "transition-all cursor-pointer"
                    )}
                  >
                    {/* Icon */}
                    <div className="w-10 h-10 rounded-lg bg-white/[0.05] flex items-center justify-center">
                      <FileIcon type={doc.file_type || doc.filename || ""} className="w-5 h-5 text-white/50" />
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-white truncate">
                        {doc.title || doc.filename}
                      </p>
                      <div className="flex items-center gap-2 text-xs text-white/40">
                        <span>{doc.file_type?.split("/").pop() || "file"}</span>
                        <span>•</span>
                        <span>{new Date(doc.created_at).toLocaleDateString()}</span>
                        {doc.status === "processing" && (
                          <>
                            <span>•</span>
                            <span className="text-violet-400">Processing...</span>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Status */}
                    <div className={cn(
                      "px-2 py-1 rounded-full text-[10px] font-medium capitalize",
                      doc.status === "ready"
                        ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                        : doc.status === "error"
                        ? "bg-red-500/10 text-red-400 border border-red-500/20"
                        : doc.status === "processing"
                        ? "bg-violet-500/10 text-violet-400 border border-violet-500/20"
                        : doc.status === "partial"
                        ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                        : "bg-white/5 text-white/40 border border-white/10"
                    )}>
                      {doc.status}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-1">
                      
                      <motion.button
                        whileHover={{ scale: 1.1 }}
                        onClick={() => handleDelete(doc.id)}
                        className="p-2 rounded-lg text-white/40 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </motion.button>
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </>
        )}
      </div>
    </main>
  );
}
