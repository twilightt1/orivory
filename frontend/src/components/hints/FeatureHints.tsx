"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";

// Types
interface HintContent {
  title: string;
  body: string;
  action_label?: string;
  action_url?: string;
  icon?: string;
}

interface Hint {
  id: string;
  feature: string;
  content: HintContent;
  priority: "high" | "medium" | "low";
}

interface HintContextValue {
  hints: Hint[];
  dismissHint: (hintId: string, reason?: string) => void;
  snoozeHint: (hintId: string, days?: number) => void;
  trackFeatureUse: (feature: string) => void;
  isLoading: boolean;
}

const HintContext = createContext<HintContextValue | null>(null);

export function useFeatureHints() {
  const context = useContext(HintContext);
  if (!context) {
    throw new Error("useFeatureHints must be used within FeatureHintProvider");
  }
  return context;
}

// Hint display component
interface HintCardProps {
  hint: Hint;
  onDismiss: () => void;
  onAction?: () => void;
  onSnooze?: () => void;
}

function HintCard({ hint, onDismiss, onAction, onSnooze }: HintCardProps) {
  return (
    <div
      data-testid={`hint-card-${hint.id}`}
      className="hint-card"
      style={{
        position: "fixed",
        bottom: "1rem",
        right: "1rem",
        maxWidth: "360px",
        backgroundColor: "white",
        borderRadius: "12px",
        boxShadow: "0 4px 20px rgba(0, 0, 0, 0.15)",
        padding: "1rem",
        zIndex: 1000,
        animation: "slideIn 0.3s ease-out",
      }}
    >
      {/* Icon and title */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: "0.75rem" }}>
        {hint.content.icon && (
          <span style={{ fontSize: "1.5rem" }}>{hint.content.icon}</span>
        )}
        <div style={{ flex: 1 }}>
          <h4
            style={{
              margin: 0,
              fontSize: "0.95rem",
              fontWeight: 600,
              color: "#1a1a2e",
            }}
          >
            {hint.content.title}
          </h4>
          <p
            style={{
              margin: "0.5rem 0 0",
              fontSize: "0.85rem",
              color: "#666",
              lineHeight: 1.4,
            }}
          >
            {hint.content.body}
          </p>
        </div>
      </div>

      {/* Actions */}
      <div
        style={{
          display: "flex",
          justifyContent: "flex-end",
          gap: "0.5rem",
          marginTop: "1rem",
        }}
      >
        <button
          onClick={onSnooze}
          style={{
            padding: "0.4rem 0.75rem",
            border: "none",
            background: "transparent",
            color: "#666",
            fontSize: "0.8rem",
            cursor: "pointer",
            borderRadius: "6px",
          }}
        >
          Later
        </button>
        {hint.content.action_url && (
          <button
            onClick={onAction}
            style={{
              padding: "0.4rem 0.75rem",
              border: "none",
              background: "#6366f1",
              color: "white",
              fontSize: "0.8rem",
              fontWeight: 500,
              cursor: "pointer",
              borderRadius: "6px",
            }}
          >
            {hint.content.action_label || "View"}
          </button>
        )}
        <button
          onClick={onDismiss}
          style={{
            padding: "0.4rem 0.75rem",
            border: "none",
            background: "#f1f1f1",
            color: "#333",
            fontSize: "0.8rem",
            cursor: "pointer",
            borderRadius: "6px",
          }}
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}

// Provider component
interface FeatureHintProviderProps {
  children: React.ReactNode;
  apiBaseUrl?: string;
  maxHints?: number;
}

export function FeatureHintProvider({
  children,
  apiBaseUrl = "",
  maxHints = 1,
}: FeatureHintProviderProps) {
  const [hints, setHints] = useState<Hint[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch available hints
  const fetchHints = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/hints`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setHints(data.hints.slice(0, maxHints));
      }
    } catch (error) {
      console.error("Failed to fetch hints:", error);
    } finally {
      setIsLoading(false);
    }
  }, [apiBaseUrl, maxHints]);

  // Record interaction
  const recordInteraction = useCallback(
    async (hintId: string, action: string) => {
      try {
        await fetch(`${apiBaseUrl}/api/v1/hints/interactions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
          },
          body: JSON.stringify({
            hint_id: hintId,
            action,
          }),
        });
      } catch (error) {
        console.error("Failed to record hint interaction:", error);
      }
    },
    [apiBaseUrl]
  );

  // Dismiss hint
  const dismissHint = useCallback(
    async (hintId: string, reason?: string) => {
      await recordInteraction(hintId, "dismissed");
      setHints((prev) => prev.filter((h) => h.id !== hintId));
    },
    [recordInteraction]
  );

  // Snooze hint
  const snoozeHint = useCallback(
    async (hintId: string, days = 7) => {
      await recordInteraction(hintId, "snoozed");
      setHints((prev) => prev.filter((h) => h.id !== hintId));
    },
    [recordInteraction]
  );

  // Track feature use
  const trackFeatureUse = useCallback(
    async (feature: string) => {
      try {
        await fetch(`${apiBaseUrl}/api/v1/hints/track-feature/${feature}`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
          },
        });
      } catch (error) {
        console.error("Failed to track feature use:", error);
      }
    },
    [apiBaseUrl]
  );

  // Handle action click
  const handleAction = useCallback(
    async (hint: Hint) => {
      await recordInteraction(hint.id, "actioned");
      if (hint.content.action_url) {
        window.location.href = hint.content.action_url;
      }
    },
    [recordInteraction]
  );

  // Initial fetch
  useEffect(() => {
    fetchHints();
  }, [fetchHints]);

  const value: HintContextValue = {
    hints,
    dismissHint,
    snoozeHint,
    trackFeatureUse,
    isLoading,
  };

  return (
    <HintContext.Provider value={value}>
      {children}
      {/* Render visible hints */}
      {hints.map((hint) => (
        <HintCard
          key={hint.id}
          hint={hint}
          onDismiss={() => dismissHint(hint.id)}
          onSnooze={() => snoozeHint(hint.id)}
          onAction={() => handleAction(hint)}
        />
      ))}
    </HintContext.Provider>
  );
}

// Hook for manually triggering hints
export function useHintTrigger() {
  const { trackFeatureUse } = useFeatureHints();

  return {
    trackDiscovery: () => trackFeatureUse("discovery"),
    trackInsights: () => trackFeatureUse("insights"),
    trackSources: () => trackFeatureUse("sources"),
    trackSearch: () => trackFeatureUse("search"),
    trackWorkspaces: () => trackFeatureUse("workspaces"),
  };
}

export default FeatureHintProvider;
