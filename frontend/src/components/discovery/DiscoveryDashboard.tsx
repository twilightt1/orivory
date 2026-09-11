"use client";

import { useState, useEffect, useCallback } from "react";
import { cn } from "@/lib/utils";
import { 
  Compass, 
  Play, 
  ChevronRight,
  Loader2,
  Map,
  TrendingUp,
  GitBranch,
  Lightbulb,
  Clock
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import type {
  DiscoverySession,
  DiscoveryStep,
  GraphMetrics,
  DiscoveryFlowType,
  GraphResponse
} from "@/types/discovery";
import { FLOW_TYPE_CONFIG } from "@/types/discovery";
import {
  listDiscoverySessions,
  getNextDiscoveryStep,
  advanceDiscovery,
  completeDiscovery,
  getGraphMetrics,
  createDiscoverySession,
} from "@/lib/api/discovery";
import { BentoGrid, BentoCard } from "@/components/ui/bento-grid";
import { Timeline } from "@/components/ui/timeline";
import { SpotlightCard } from "@/components/ui/spotlight";
import { StatCard } from "@/components/ui/stats";
import { PageSkeleton } from "@/components/ui/skeleton";
import { KnowledgeGraphVisualization } from "@/components/discovery/KnowledgeGraphVisualization";
import { getDocumentGraph } from "@/lib/api/discovery";

interface DiscoveryDashboardProps {
  className?: string;
}

export function DiscoveryDashboard({ className }: DiscoveryDashboardProps) {
  const [sessions, setSessions] = useState<DiscoverySession[]>([]);
  const [metrics, setMetrics] = useState<GraphMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeSession, setActiveSession] = useState<DiscoverySession | null>(null);
  const [currentStep, setCurrentStep] = useState<DiscoveryStep | null>(null);
  const [isAdvancing, setIsAdvancing] = useState(false);
  const [showNewSessionModal, setShowNewSessionModal] = useState(false);
  const [graphData, setGraphData] = useState<GraphResponse | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  // Starting-document picker for the new-session modal. The backend requires
  // starting_doc_id to be a real UUID — the previous literal "default" was
  // rejected with 422 on every attempt, silently (console.error only).
  const [startingDocId, setStartingDocId] = useState<string>("");
  const [sessionError, setSessionError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [sessionsData, metricsData] = await Promise.all([
        listDiscoverySessions({ limit: 10 }),
        getGraphMetrics(),
      ]);
      setSessions(sessionsData.items);
      setMetrics(metricsData);
    } catch (err) {
      console.error("Failed to fetch discovery data:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchGraphData = useCallback(async () => {
    setGraphLoading(true);
    try {
      const data = await getDocumentGraph();
      setGraphData(data);
    } catch (err) {
      console.error("Failed to fetch graph data:", err);
    } finally {
      setGraphLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGraphData();
  }, [fetchGraphData]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  async function handleAdvance() {
    if (!activeSession) return;
    
    setIsAdvancing(true);
    try {
      const session = await advanceDiscovery(activeSession.id);
      setActiveSession(session);
      
      const step = await getNextDiscoveryStep(activeSession.id);
      setCurrentStep(step);
    } catch (err) {
      console.error("Failed to advance:", err);
    } finally {
      setIsAdvancing(false);
    }
  }

  async function handleComplete() {
    if (!activeSession) return;
    
    setIsAdvancing(true);
    try {
      const session = await completeDiscovery(activeSession.id);
      setActiveSession(session);
      setCurrentStep(null);
      await fetchData();
    } catch (err) {
      console.error("Failed to complete:", err);
    } finally {
      setIsAdvancing(false);
    }
  }

  async function handleStartSession(flowType: DiscoveryFlowType, startingDocId: string) {
    if (!startingDocId) {
      setSessionError("Pick a starting document first — journeys begin from a real memory.");
      return;
    }
    setSessionError(null);
    try {
      const session = await createDiscoverySession({
        flow_type: flowType,
        starting_doc_id: startingDocId,
      });
      setActiveSession(session);
      const step = await getNextDiscoveryStep(session.id);
      setCurrentStep(step);
      setShowNewSessionModal(false);
    } catch (err) {
      setSessionError(err instanceof Error ? err.message : "Failed to start session.");
    }
  }

  function handleExitSession() {
    setActiveSession(null);
    setCurrentStep(null);
  }

  if (loading) {
    return <PageSkeleton />;
  }

  // Active discovery session view
  if (activeSession) {
    return (
      <div className={cn("space-y-6", className)}>
        {/* Session Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Compass className="w-6 h-6 text-primary" />
            <div>
              <h2 className="text-xl font-bold">Discovery Journey</h2>
              <p className="text-sm text-muted-foreground">
                {FLOW_TYPE_CONFIG[activeSession.flow_type as DiscoveryFlowType]?.label}
              </p>
            </div>
          </div>
          <button
            onClick={handleExitSession}
            className="px-4 py-2 rounded-lg border border-border hover:bg-accent transition-colors text-sm"
          >
            Exit Journey
          </button>
        </div>

        {/* Progress */}
        <div className="flex items-center gap-4">
          <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-primary"
              initial={{ width: 0 }}
              animate={{ width: `${(activeSession.steps_taken / 5) * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <span className="text-sm text-muted-foreground">
            Step {activeSession.steps_taken + 1}
          </span>
        </div>

        {/* Current Step */}
        <AnimatePresence mode="wait">
          {currentStep && !currentStep.is_complete ? (
            <motion.div
              key="step"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <SpotlightCard className="rounded-2xl p-6">
                <div className="relative z-10">
                  <div className="flex items-center gap-2 mb-4">
                    <Map className="w-5 h-5 text-primary" />
                    <span className="text-sm font-medium text-primary">
                      {currentStep.step_goal}
                    </span>
                  </div>
                  
                  <h3 className="text-lg font-semibold mb-2">
                    {currentStep.next_doc_title}
                  </h3>
                  
                  <p className="text-muted-foreground mb-4">
                    {currentStep.reasoning}
                  </p>
                  
                  {currentStep.insight_preview && (
                    <div className="p-4 bg-primary/5 rounded-lg border border-primary/20">
                      <div className="flex items-center gap-2 mb-2">
                        <Lightbulb className="w-4 h-4 text-primary" />
                        <span className="text-sm font-medium text-primary">Insight Preview</span>
                      </div>
                      <p className="text-sm">{currentStep.insight_preview}</p>
                    </div>
                  )}
                </div>
              </SpotlightCard>
            </motion.div>
          ) : currentStep?.is_complete ? (
            <motion.div
              key="complete"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-center py-12"
            >
              <div className="inline-flex items-center justify-center w-16 h-16 bg-primary/20 rounded-full mb-4">
                <Lightbulb className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-2">Journey Complete!</h3>
              <p className="text-muted-foreground mb-6">
                You've explored {activeSession.documents_explored} documents and found {activeSession.connections_found.length} connections.
              </p>
            </motion.div>
          ) : null}
        </AnimatePresence>

        {/* Action Buttons */}
        <div className="flex justify-center gap-4">
          {currentStep && !currentStep.is_complete ? (
            <button
              onClick={handleAdvance}
              disabled={isAdvancing}
              className="btn-premium px-6 py-3 rounded-xl font-medium flex items-center gap-2"
            >
              {isAdvancing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
              Continue Discovery
            </button>
          ) : activeSession.status === "active" ? (
            <button
              onClick={handleComplete}
              disabled={isAdvancing}
              className="btn-premium px-6 py-3 rounded-xl font-medium flex items-center gap-2"
            >
              Complete Journey
            </button>
          ) : null}
        </div>

        {/* Connections Found */}
        {activeSession.connections_found.length > 0 && (
          <div>
            <h4 className="font-medium mb-4 flex items-center gap-2">
              <GitBranch className="w-4 h-4" />
              Connections Found
            </h4>
            <Timeline
              entries={activeSession.connections_found.map((conn, i) => ({
                title: conn.title || `Connection ${i + 1}`,
                content: conn.description || conn.insight || "",
                date: undefined,
              }))}
            />
          </div>
        )}
      </div>
    );
  }

  // Dashboard view
  return (
    <div className={cn("space-y-8", className)}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Compass className="w-6 h-6 text-primary" />
            Discovery
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Explore connections across your knowledge
          </p>
        </div>
        <button
          onClick={() => setShowNewSessionModal(true)}
          className="btn-premium px-4 py-2.5 rounded-lg font-medium flex items-center gap-2"
        >
          <Play className="w-4 h-4" />
          Start Journey
        </button>
      </div>

      {/* Stats */}
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            label="Documents"
            value={metrics.total_nodes}
            icon={<Map className="w-5 h-5" />}
          />
          <StatCard
            label="Connections"
            value={metrics.total_edges}
            icon={<GitBranch className="w-5 h-5" />}
          />
          <StatCard
            label="Avg. Connections"
            value={metrics.avg_connections_per_node.toFixed(1)}
            icon={<TrendingUp className="w-5 h-5" />}
          />
          <StatCard
            label="Graph Density"
            value={`${(metrics.graph_density * 100).toFixed(0)}%`}
            icon={<Clock className="w-5 h-5" />}
          />
        </div>
      )}

      {/* Knowledge Graph Visualization */}
      <div>
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <GitBranch className="w-5 h-5" />
          Knowledge Graph
        </h3>
        <KnowledgeGraphVisualization
          graphData={graphData}
          loading={graphLoading}
          onNodeClick={(node) => {
            console.log("Node clicked:", node);
          }}
          className="h-[400px] border border-border rounded-xl"
        />
      </div>

      {/* Flow Types */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Discovery Flows</h3>
        <BentoGrid>
          {(Object.entries(FLOW_TYPE_CONFIG) as [DiscoveryFlowType, typeof FLOW_TYPE_CONFIG[DiscoveryFlowType]][]).map(([key, config]) => (
            <BentoCard
              key={key}
              icon={<span className="text-xl">{config.emoji}</span>}
              title={config.label}
              description={config.description}
              className="cursor-pointer hover:border-primary/50"
              onClick={() => setShowNewSessionModal(true)}
            />
          ))}
        </BentoGrid>
      </div>

      {/* Recent Sessions */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Recent Journeys</h3>
        {sessions.length === 0 ? (
          <div className="text-center py-12 border border-dashed border-border rounded-xl">
            <Compass className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <h4 className="font-medium mb-2">No journeys yet</h4>
            <p className="text-sm text-muted-foreground mb-4">
              Start a discovery journey to explore connections in your knowledge.
            </p>
            <button
              onClick={() => setShowNewSessionModal(true)}
              className="btn-premium px-4 py-2 rounded-lg text-sm font-medium"
            >
              Start Your First Journey
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {sessions.map((session) => (
              <button
                key={session.id}
                onClick={() => {
                  setActiveSession(session);
                  getNextDiscoveryStep(session.id).then(setCurrentStep);
                }}
                className="w-full p-4 bg-card border border-border rounded-xl text-left hover:border-border/80 transition-colors group"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">
                      {FLOW_TYPE_CONFIG[session.flow_type as DiscoveryFlowType]?.emoji}
                    </span>
                    <div>
                      <h4 className="font-medium">
                        {FLOW_TYPE_CONFIG[session.flow_type as DiscoveryFlowType]?.label}
                      </h4>
                      <p className="text-sm text-muted-foreground">
                        {session.documents_explored} documents • {session.connections_found.length} connections
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        "px-2 py-1 rounded-full text-xs font-medium",
                        session.status === "completed"
                          ? "bg-green-500/20 text-green-500"
                          : "bg-primary/20 text-primary"
                      )}
                    >
                      {session.status}
                    </span>
                    <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors" />
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* New Session Modal */}
      <AnimatePresence>
        {showNewSessionModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
          >
            <div
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
              onClick={() => setShowNewSessionModal(false)}
            />
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="relative bg-card rounded-2xl border border-border p-6 w-full max-w-lg shadow-2xl"
            >
              <h3 className="text-xl font-bold mb-4">Start a Discovery Journey</h3>
              <p className="text-muted-foreground mb-4">
                Pick a starting memory, then select a flow type to begin exploring connections in your knowledge.
              </p>
              <label className="block text-sm font-medium mb-1" htmlFor="discovery-start-doc">
                Starting memory
              </label>
              <select
                id="discovery-start-doc"
                value={startingDocId}
                onChange={(e) => setStartingDocId(e.target.value)}
                className="w-full mb-4 p-2.5 bg-card border border-border rounded-xl text-sm"
              >
                <option value="">Select a memory…</option>
                {(graphData?.nodes ?? []).slice(0, 100).map((n) => (
                  <option key={n.id} value={n.id}>
                    {n.title || n.id}
                  </option>
                ))}
              </select>
              {sessionError && (
                <p role="alert" className="mb-4 text-sm text-destructive">
                  {sessionError}
                </p>
              )}
              <div className="space-y-3">
                {(Object.entries(FLOW_TYPE_CONFIG) as [DiscoveryFlowType, typeof FLOW_TYPE_CONFIG[DiscoveryFlowType]][]).map(([key, config]) => (
                  <button
                    key={key}
                    onClick={() => handleStartSession(key, startingDocId)}
                    className="w-full p-4 bg-card border border-border rounded-xl text-left hover:border-primary/50 hover:bg-accent/50 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{config.emoji}</span>
                      <div>
                        <h4 className="font-medium">{config.label}</h4>
                        <p className="text-sm text-muted-foreground">{config.description}</p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
              <button
                onClick={() => setShowNewSessionModal(false)}
                className="mt-4 w-full py-2 text-center text-sm text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
