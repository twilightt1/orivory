"use client";

import { useState, useEffect, useCallback } from "react";
import { cn } from "@/lib/utils";
import { 
  Users, 
  Plus, 
  Settings, 
  Loader2,
  Crown,
  Shield,
  Pencil,
  Eye,
  Mail,
  Copy,
  Check,
  X,
  MoreHorizontal,
  LogOut,
  UserPlus,
  RefreshCw
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import type { 
  Workspace, 
  TeamMember, 
  WorkspaceInvite,
  MemberRole,
  WorkspaceCreate,
  WorkspaceUpdate,
  InviteCreate 
} from "@/types/workspace";
import { ROLE_CONFIG, WORKSPACE_TYPE_CONFIG } from "@/types/workspace";
import {
  listWorkspaces,
  createWorkspace,
  getWorkspace,
  updateWorkspace,
  deleteWorkspace,
  listMembers,
  addMember,
  updateMemberRole,
  removeMember,
  listInvites,
  createInvite,
  cancelInvite,
} from "@/lib/api/workspaces";
import { BentoGrid, BentoCard } from "@/components/ui/bento-grid";
import { Modal, ModalHeader, ModalBody, ModalFooter } from "@/components/ui/modal";
import { StatCard } from "@/components/ui/stats";
import { ExpandableCard } from "@/components/ui/card-stack";

interface WorkspacesDashboardProps {
  className?: string;
}

export function WorkspacesDashboard({ className }: WorkspacesDashboardProps) {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [selectedWorkspace, setSelectedWorkspace] = useState<Workspace | null>(null);
  // The detail effect (below) is keyed on this id, NOT on the workspace
  // object: every fetch returns a fresh object identity, so depending on the
  // object re-fires the effect forever (unbounded refetch loop).
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState<string | null>(null);
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [invites, setInvites] = useState<WorkspaceInvite[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"overview" | "members" | "invites">("overview");
  
  // Modals
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showInviteModal, setShowInviteModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);

  const fetchWorkspaces = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listWorkspaces();
      setWorkspaces(data.items);
    } catch (err) {
      console.error("Failed to fetch workspaces:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchWorkspaceDetails = useCallback(async (workspaceId: string) => {
    try {
      const [workspaceData, membersData, invitesData] = await Promise.all([
        getWorkspace(workspaceId),
        listMembers(workspaceId),
        listInvites(workspaceId),
      ]);
      setSelectedWorkspace(workspaceData);
      setMembers(membersData.items);
      setInvites(invitesData.items);
    } catch (err) {
      console.error("Failed to fetch workspace details:", err);
    }
  }, []);

  useEffect(() => {
    fetchWorkspaces();
  }, [fetchWorkspaces]);

  useEffect(() => {
    if (selectedWorkspaceId) {
      fetchWorkspaceDetails(selectedWorkspaceId);
    }
  }, [selectedWorkspaceId, fetchWorkspaceDetails]);

  async function handleCreateWorkspace(data: WorkspaceCreate) {
    try {
      const workspace = await createWorkspace(data);
      setWorkspaces([workspace, ...workspaces]);
      setShowCreateModal(false);
      setSelectedWorkspace(workspace);
      setSelectedWorkspaceId(workspace.id);
    } catch (err) {
      console.error("Failed to create workspace:", err);
    }
  }

  async function handleInvite(data: InviteCreate) {
    if (!selectedWorkspace) return;
    try {
      const invite = await createInvite(selectedWorkspace.id, data);
      setInvites([...invites, invite]);
      setShowInviteModal(false);
    } catch (err) {
      console.error("Failed to create invite:", err);
    }
  }

  async function handleUpdateMemberRole(userId: string, role: MemberRole) {
    if (!selectedWorkspace) return;
    try {
      const updated = await updateMemberRole(selectedWorkspace.id, userId, role);
      setMembers(members.map(m => m.user_id === userId ? updated : m));
    } catch (err) {
      console.error("Failed to update member role:", err);
    }
  }

  async function handleRemoveMember(userId: string) {
    if (!selectedWorkspace) return;
    try {
      await removeMember(selectedWorkspace.id, userId);
      setMembers(members.filter(m => m.user_id !== userId));
    } catch (err) {
      console.error("Failed to remove member:", err);
    }
  }

  async function handleCancelInvite(inviteId: string) {
    if (!selectedWorkspace) return;
    try {
      await cancelInvite(selectedWorkspace.id, inviteId);
      setInvites(invites.filter(i => i.id !== inviteId));
    } catch (err) {
      console.error("Failed to cancel invite:", err);
    }
  }

  if (loading) {
    return (
      <div className={cn("flex items-center justify-center py-20", className)}>
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // Workspace detail view
  if (selectedWorkspace) {
    const roleIcons: Record<MemberRole, React.ReactNode> = {
      owner: <Crown className="w-4 h-4 text-amber-400" />,
      admin: <Shield className="w-4 h-4 text-purple-400" />,
      editor: <Pencil className="w-4 h-4 text-blue-400" />,
      viewer: <Eye className="w-4 h-4 text-green-400" />,
    };

    return (
      <div className={cn("space-y-6", className)}>
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => { setSelectedWorkspace(null); setSelectedWorkspaceId(null); }}
              className="p-2 rounded-lg hover:bg-accent transition-colors"
            >
              ←
            </button>
            <div>
              <h2 className="text-xl font-bold flex items-center gap-2">
                {selectedWorkspace.name}
                <span className="text-sm text-muted-foreground font-normal">
                  {WORKSPACE_TYPE_CONFIG[selectedWorkspace.workspace_type]?.emoji}
                </span>
              </h2>
              {selectedWorkspace.description && (
                <p className="text-sm text-muted-foreground">{selectedWorkspace.description}</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowInviteModal(true)}
              className="btn-premium px-4 py-2 rounded-lg font-medium flex items-center gap-2 text-sm"
            >
              <UserPlus className="w-4 h-4" />
              Invite
            </button>
            <button
              onClick={() => setShowSettingsModal(true)}
              className="p-2 rounded-lg border border-border hover:bg-accent transition-colors"
            >
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 p-1 bg-muted rounded-lg w-fit">
          {(["overview", "members", "invites"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={cn(
                "px-4 py-2 rounded-md text-sm font-medium transition-colors capitalize",
                activeTab === tab
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                label="Members"
                value={selectedWorkspace.member_count}
                icon={<Users className="w-5 h-5" />}
              />
              <StatCard
                label="Documents"
                value={0}
                icon={<Plus className="w-5 h-5" />}
              />
            </div>
          </div>
        )}

        {/* Members Tab */}
        {activeTab === "members" && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Team Members ({members.length})</h3>
              <button
                onClick={() => setShowInviteModal(true)}
                className="text-sm text-primary hover:underline"
              >
                + Add member
              </button>
            </div>
            
            <div className="space-y-2">
              {members.map((member) => (
                <div
                  key={member.user_id}
                  className="flex items-center justify-between p-4 bg-card border border-border rounded-xl"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                      <Users className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">
                          {member.user_id.slice(0, 8)}... (User ID)
                        </span>
                        {roleIcons[member.role]}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Joined {new Date(member.joined_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  
                  {member.role !== "owner" && (
                    <div className="flex items-center gap-2">
                      <select
                        value={member.role}
                        onChange={(e) => handleUpdateMemberRole(member.user_id, e.target.value as MemberRole)}
                        className="bg-transparent border border-border rounded-lg px-2 py-1 text-sm"
                      >
                        <option value="admin">Admin</option>
                        <option value="editor">Editor</option>
                        <option value="viewer">Viewer</option>
                      </select>
                      <button
                        onClick={() => handleRemoveMember(member.user_id)}
                        className="p-2 rounded-lg hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                      >
                        <LogOut className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Invites Tab */}
        {activeTab === "invites" && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Pending Invites ({invites.length})</h3>
              <button
                onClick={() => setShowInviteModal(true)}
                className="text-sm text-primary hover:underline"
              >
                + New invite
              </button>
            </div>
            
            {invites.length === 0 ? (
              <div className="text-center py-8 border border-dashed border-border rounded-xl">
                <Mail className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                <p className="text-sm text-muted-foreground">No pending invites</p>
              </div>
            ) : (
              <div className="space-y-2">
                {invites.map((invite) => (
                  <div
                    key={invite.id}
                    className="flex items-center justify-between p-4 bg-card border border-border rounded-xl"
                  >
                    <div className="flex items-center gap-3">
                      <Mail className="w-5 h-5 text-muted-foreground" />
                      <div>
                        <span className="font-medium">{invite.email}</span>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <span>as {invite.role}</span>
                          <span>•</span>
                          <span>Expires {new Date(invite.expires_at).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => handleCancelInvite(invite.id)}
                      className="p-2 rounded-lg hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Invite Modal */}
        <InviteModal
          isOpen={showInviteModal}
          onClose={() => setShowInviteModal(false)}
          onSubmit={handleInvite}
        />

        {/* Settings Modal */}
        <SettingsModal
          workspace={selectedWorkspace}
          isOpen={showSettingsModal}
          onClose={() => setShowSettingsModal(false)}
          onSave={async (data) => {
            if (selectedWorkspace) {
              const updated = await updateWorkspace(selectedWorkspace.id, data);
              setSelectedWorkspace(updated);
              setWorkspaces(workspaces.map(w => w.id === updated.id ? updated : w));
            }
          }}
          onDelete={async () => {
            if (selectedWorkspace) {
              await deleteWorkspace(selectedWorkspace.id);
              setWorkspaces(workspaces.filter(w => w.id !== selectedWorkspace.id));
              setSelectedWorkspace(null);
              setSelectedWorkspaceId(null);
            }
          }}
        />
      </div>
    );
  }

  // Workspaces list view
  return (
    <div className={cn("space-y-8", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Users className="w-6 h-6 text-primary" />
            Workspaces
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Manage your team knowledge bases
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn-premium px-4 py-2.5 rounded-lg font-medium flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Create Workspace
        </button>
      </div>

      {/* Workspaces Grid */}
      {workspaces.length === 0 ? (
        <div className="text-center py-16 border border-dashed border-border rounded-2xl">
          <Users className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium mb-2">No workspaces yet</h3>
          <p className="text-muted-foreground mb-6 max-w-md mx-auto">
            Create a workspace to start collaborating with your team on shared knowledge.
          </p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn-premium px-6 py-3 rounded-xl font-medium"
          >
            Create Your First Workspace
          </button>
        </div>
      ) : (
        <BentoGrid>
          {workspaces.map((workspace) => (
            <BentoCard
              key={workspace.id}
              title={workspace.name}
              description={workspace.description || `${workspace.member_count} members`}
              icon={<Users className="w-5 h-5 text-primary" />}
              className="cursor-pointer"
              onClick={() => { setSelectedWorkspace(workspace); setSelectedWorkspaceId(workspace.id); }}
            />
          ))}
        </BentoGrid>
      )}

      {/* Create Modal */}
      <CreateWorkspaceModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onSubmit={handleCreateWorkspace}
      />
    </div>
  );
}

// ─── Sub-components ────────────────────────────────────────────────────────────

function CreateWorkspaceModal({
  isOpen,
  onClose,
  onSubmit,
}: {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: WorkspaceCreate) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [workspaceType, setWorkspaceType] = useState<"personal" | "team">("team");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    
    setIsSubmitting(true);
    try {
      await onSubmit({
        name: name.trim(),
        description: description.trim() || undefined,
        workspace_type: workspaceType,
      });
      setName("");
      setDescription("");
      setWorkspaceType("team");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <form onSubmit={handleSubmit}>
        <ModalHeader>
          <h3 className="text-xl font-bold">Create Workspace</h3>
          <p className="text-sm text-muted-foreground mt-1">
            Set up a space for your team to share knowledge.
          </p>
        </ModalHeader>
        <ModalBody>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1.5">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Research Team"
                className="w-full px-4 py-2.5 bg-background border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1.5">Description</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Our team's shared knowledge base..."
                className="w-full px-4 py-2.5 bg-background border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
                rows={3}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1.5">Type</label>
              <div className="flex gap-3">
                {(["team", "personal"] as const).map((type) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => setWorkspaceType(type)}
                    className={cn(
                      "flex-1 p-3 rounded-xl border text-left transition-colors",
                      workspaceType === type
                        ? "border-primary bg-primary/5"
                        : "border-border hover:border-border/80"
                    )}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-lg">{WORKSPACE_TYPE_CONFIG[type].emoji}</span>
                      <span className="font-medium">{WORKSPACE_TYPE_CONFIG[type].label}</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {type === "team" ? "Collaborate with your team" : "Personal workspace"}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </ModalBody>
        <ModalFooter>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg border border-border hover:bg-accent transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!name.trim() || isSubmitting}
            className="btn-premium px-4 py-2 rounded-lg font-medium flex items-center gap-2 disabled:opacity-50"
          >
            {isSubmitting ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
            Create Workspace
          </button>
        </ModalFooter>
      </form>
    </Modal>
  );
}

function InviteModal({
  isOpen,
  onClose,
  onSubmit,
}: {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: InviteCreate) => void;
}) {
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<MemberRole>("viewer");
  const [message, setMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!email.trim()) return;
    
    setIsSubmitting(true);
    try {
      await onSubmit({
        email: email.trim(),
        role,
        message: message.trim() || undefined,
      });
      setEmail("");
      setMessage("");
      setRole("viewer");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <form onSubmit={handleSubmit}>
        <ModalHeader>
          <h3 className="text-xl font-bold">Invite Member</h3>
          <p className="text-sm text-muted-foreground mt-1">
            Send an invitation to join your workspace.
          </p>
        </ModalHeader>
        <ModalBody>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1.5">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="colleague@company.com"
                className="w-full px-4 py-2.5 bg-background border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1.5">Role</label>
              <div className="grid grid-cols-3 gap-2">
                {(["admin", "editor", "viewer"] as MemberRole[]).map((r) => (
                  <button
                    key={r}
                    type="button"
                    onClick={() => setRole(r)}
                    className={cn(
                      "p-3 rounded-xl border text-center transition-colors",
                      role === r
                        ? "border-primary bg-primary/5"
                        : "border-border hover:border-border/80"
                    )}
                  >
                    <span className="text-lg">{r === "admin" ? "🛡️" : r === "editor" ? "✏️" : "👁️"}</span>
                    <div className="text-sm font-medium mt-1 capitalize">{r}</div>
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1.5">Message (optional)</label>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Join our team workspace to collaborate on..."
                className="w-full px-4 py-2.5 bg-background border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
                rows={2}
              />
            </div>
          </div>
        </ModalBody>
        <ModalFooter>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg border border-border hover:bg-accent transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!email.trim() || isSubmitting}
            className="btn-premium px-4 py-2 rounded-lg font-medium flex items-center gap-2 disabled:opacity-50"
          >
            {isSubmitting ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
            Send Invite
          </button>
        </ModalFooter>
      </form>
    </Modal>
  );
}

function SettingsModal({
  workspace,
  isOpen,
  onClose,
  onSave,
  onDelete,
}: {
  workspace: Workspace;
  isOpen: boolean;
  onClose: () => void;
  onSave: (data: WorkspaceUpdate) => void;
  onDelete: () => void;
}) {
  const [name, setName] = useState(workspace.name);
  const [description, setDescription] = useState(workspace.description || "");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    setName(workspace.name);
    setDescription(workspace.description || "");
  }, [workspace]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      await onSave({
        name: name.trim(),
        description: description.trim() || undefined,
      });
      onClose();
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <form onSubmit={handleSubmit}>
        <ModalHeader>
          <h3 className="text-xl font-bold">Workspace Settings</h3>
        </ModalHeader>
        <ModalBody>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1.5">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-4 py-2.5 bg-background border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1.5">Description</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="w-full px-4 py-2.5 bg-background border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
                rows={3}
              />
            </div>
            <div className="pt-4 border-t border-border">
              <h4 className="font-medium text-destructive mb-2">Danger Zone</h4>
              <button
                type="button"
                onClick={onDelete}
                className="w-full p-3 rounded-xl border border-destructive/30 text-destructive hover:bg-destructive/10 transition-colors text-left"
              >
                <div className="flex items-center gap-2">
                  <X className="w-4 h-4" />
                  <span className="font-medium">Delete Workspace</span>
                </div>
                <p className="text-sm text-muted-foreground mt-1">
                  This action cannot be undone. All data will be permanently deleted.
                </p>
              </button>
            </div>
          </div>
        </ModalBody>
        <ModalFooter>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg border border-border hover:bg-accent transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!name.trim() || isSubmitting}
            className="btn-premium px-4 py-2 rounded-lg font-medium disabled:opacity-50"
          >
            Save Changes
          </button>
        </ModalFooter>
      </form>
    </Modal>
  );
}
