"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Eye,
  EyeOff,
  GitBranch,
  Loader2,
  Info,
  X
} from "lucide-react";
import type { GraphResponse, DocumentNode, RelationshipEdge } from "@/types/discovery";

interface KnowledgeGraphVisualizationProps {
  graphData: GraphResponse | null;
  loading?: boolean;
  onNodeClick?: (node: DocumentNode) => void;
  onEdgeClick?: (edge: RelationshipEdge) => void;
  className?: string;
  maxNodes?: number;
}

// Color palette matching Orivory's design system
const NODE_COLORS: Record<string, string> = {
  person: "#a78bfa",
  project: "#60a5fa",
  topic: "#34d399",
  concept: "#fbbf24",
  organization: "#f472b6",
  place: "#38bdf8",
  date: "#94a3b8",
  event: "#fb923c",
  media: "#c084fc",
  other: "#9ca3af",
};

const DEFAULT_NODE_COLOR = "#6b7280";
const EDGE_COLOR = "rgba(255, 255, 255, 0.15)";
const EDGE_HOVER_COLOR = "rgba(139, 92, 246, 0.6)";

interface LayoutNode extends DocumentNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

interface LayoutLink {
  source: string;
  target: string;
  relationship_type: string;
  weight: number;
  evidence: string;
}

export function KnowledgeGraphVisualization({
  graphData,
  loading = false,
  onNodeClick,
  className,
  maxNodes = 100,
}: KnowledgeGraphVisualizationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);
  const [nodes, setNodes] = useState<LayoutNode[]>([]);
  const [links, setLinks] = useState<LayoutLink[]>([]);
  const animFrameRef = useRef<number | null>(null);

  // Get unique entity types
  const entityTypes = graphData
    ? Array.from(new Set(graphData.nodes.flatMap(n => n.entity_ids || [])))
    : [];

  // Simple force-directed layout simulation.
  // Pure function of its inputs: the caller (graphData effect) runs it ONCE
  // per dataset. It must NOT read component state — the previous version
  // closed over `nodes` and called setNodes with a fresh array, which changed
  // the callback identity and re-fired the trigger effect forever (infinite
  // O(n^2) loop freezing the tab).
  const runSimulation = useCallback((
    inputNodes: LayoutNode[],
    inputLinks: LayoutLink[],
    viewport: { width: number; height: number },
  ): LayoutNode[] => {
    if (!inputNodes.length || !viewport.width) return inputNodes;

    const width = viewport.width;
    const height = viewport.height;
    const centerX = width / 2;
    const centerY = height / 2;

    // Initialize positions if needed
    const simNodes = inputNodes.map(n => ({
      ...n,
      x: n.x || centerX + (Math.random() - 0.5) * 200,
      y: n.y || centerY + (Math.random() - 0.5) * 200,
      vx: 0,
      vy: 0,
    }));

    const simLinks = inputLinks.map(l => ({ ...l }));

    // Run simulation iterations
    const alpha = 0.1;
    const repulsionStrength = 5000;
    const attractionStrength = 0.01;
    const centeringStrength = 0.01;
    const damping = 0.9;

    for (let i = 0; i < 100; i++) {
      // Repulsion between all nodes
      for (let a = 0; a < simNodes.length; a++) {
        for (let b = a + 1; b < simNodes.length; b++) {
          const dx = simNodes[b].x - simNodes[a].x;
          const dy = simNodes[b].y - simNodes[a].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = repulsionStrength / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          simNodes[a].vx -= fx * alpha;
          simNodes[a].vy -= fy * alpha;
          simNodes[b].vx += fx * alpha;
          simNodes[b].vy += fy * alpha;
        }
      }

      // Attraction along links
      for (const link of simLinks) {
        const source = simNodes.find(n => n.id === link.source);
        const target = simNodes.find(n => n.id === link.target);
        if (!source || !target) continue;

        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (dist - 100) * attractionStrength;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        source.vx += fx * alpha;
        source.vy += fy * alpha;
        target.vx -= fx * alpha;
        target.vy -= fy * alpha;
      }

      // Centering force
      for (const node of simNodes) {
        node.vx += (centerX - node.x) * centeringStrength * alpha;
        node.vy += (centerY - node.y) * centeringStrength * alpha;
      }

      // Apply velocities
      for (const node of simNodes) {
        node.x += node.vx;
        node.y += node.vy;
        node.vx *= damping;
        node.vy *= damping;

        // Boundary constraints
        node.x = Math.max(40, Math.min(width - 40, node.x));
        node.y = Math.max(40, Math.min(height - 40, node.y));
      }
    }

    return simNodes;
  }, []);

  // Initialize nodes and links from graphData
  useEffect(() => {
    if (!graphData) {
      setNodes([]);
      setLinks([]);
      return;
    }

    const filteredNodes = graphData.nodes.slice(0, maxNodes).map(n => ({
      ...n,
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
    }));

    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredLinks = graphData.edges
      .filter(e => nodeIds.has(e.source_id) && nodeIds.has(e.target_id))
      .slice(0, maxNodes * 2)
      .map(e => ({
        source: e.source_id,
        target: e.target_id,
        relationship_type: e.relationship_type,
        weight: e.weight,
        evidence: e.evidence,
      }));

    setNodes(runSimulation(filteredNodes, filteredLinks, dimensions));
    setLinks(filteredLinks);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphData, maxNodes]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({ width: rect.width, height: Math.max(400, rect.height) });
      }
    };

    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !nodes.length) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = dimensions.width * dpr;
    canvas.height = dimensions.height * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.clearRect(0, 0, dimensions.width, dimensions.height);

    // Draw links
    for (const link of links) {
      const source = nodes.find(n => n.id === link.source);
      const target = nodes.find(n => n.id === link.target);
      if (!source || !target) continue;

      const isHovered = hoveredNodeId === link.source || hoveredNodeId === link.target;

      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = isHovered ? EDGE_HOVER_COLOR : EDGE_COLOR;
      ctx.lineWidth = Math.max(1, link.weight * 3);
      ctx.globalAlpha = isHovered ? 0.8 : 0.4;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Draw nodes
    for (const node of nodes) {
      const color = (node.entity_ids?.[0] && NODE_COLORS[node.entity_ids[0]]) || DEFAULT_NODE_COLOR;
      const isHovered = hoveredNodeId === node.id;
      const isSelected = selectedNode === node.id;
      const radius = isHovered ? 14 : isSelected ? 16 : 10;

      // Glow for selected
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius + 8, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.2;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      // Node circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      if (showLabels) {
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillStyle = "rgba(255,255,255,0.8)";
        ctx.textAlign = "left";
        ctx.fillText(
          node.title.length > 18 ? node.title.slice(0, 18) + "..." : node.title,
          node.x + 16,
          node.y + 4
        );
      }
    }
  }, [nodes, links, dimensions, hoveredNodeId, selectedNode, showLabels]);

  // Mouse interactions
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    let foundNode: string | null = null;
    for (const node of nodes) {
      const dx = x - node.x;
      const dy = y - node.y;
      if (Math.sqrt(dx * dx + dy * dy) < 16) {
        foundNode = node.id;
        break;
      }
    }

    setHoveredNodeId(foundNode);
    canvas.style.cursor = foundNode ? "pointer" : "default";

    if (foundNode) {
      setTooltip({
        x: e.clientX,
        y: e.clientY,
        content: nodes.find(n => n.id === foundNode)?.title || "",
      });
    } else {
      setTooltip(null);
    }
  }, [nodes]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    for (const node of nodes) {
      const dx = x - node.x;
      const dy = y - node.y;
      if (Math.sqrt(dx * dx + dy * dy) < 16) {
        setSelectedNode(prev => prev === node.id ? null : node.id);
        onNodeClick?.(node);
        break;
      }
    }
  }, [nodes, onNodeClick]);

  // Selected node details
  const selectedNodeData = selectedNode && graphData
    ? graphData.nodes.find(n => n.id === selectedNode)
    : null;

  const selectedNodeEdges = selectedNode && graphData
    ? graphData.edges.filter(e => e.source_id === selectedNode || e.target_id === selectedNode)
    : [];

  return (
    <div ref={containerRef} className={cn("relative rounded-xl overflow-hidden bg-white/[0.02]", className)}>
      {/* Controls */}
      <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
        <div className="flex items-center gap-1 p-1 bg-black/40 backdrop-blur-sm rounded-lg border border-white/10">
          <button
            onClick={() => setNodes(runSimulation(nodes, links, dimensions))}
            className="p-2 hover:bg-white/10 rounded transition-colors"
            title="Re-layout"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowLabels(!showLabels)}
            className={cn(
              "p-2 rounded transition-colors",
              showLabels ? "bg-violet-500/30" : "hover:bg-white/10"
            )}
            title={showLabels ? "Hide labels" : "Show labels"}
          >
            {showLabels ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-10 p-3 bg-black/40 backdrop-blur-sm rounded-lg border border-white/10">
        <div className="flex items-center gap-1 text-xs text-white/60 mb-2">
          <Info className="w-3 h-3" />
          Legend
        </div>
        <div className="flex flex-wrap gap-3">
          {Object.entries(NODE_COLORS).slice(0, 6).map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs text-white/60 capitalize">{type}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Stats */}
      {graphData && (
        <div className="absolute top-4 left-4 z-10 px-3 py-2 bg-black/40 backdrop-blur-sm rounded-lg border border-white/10 flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-violet-400" />
            <span className="text-xs text-white/60">{nodes.length} nodes</span>
          </div>
          <div className="flex items-center gap-1.5">
            <GitBranch className="w-3 h-3 text-white/40" />
            <span className="text-xs text-white/60">{links.length} edges</span>
          </div>
        </div>
      )}

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        className="w-full h-full bg-gradient-to-br from-black/20 via-transparent to-purple-900/10"
        style={{ cursor: "default" }}
        onMouseMove={handleMouseMove}
        onClick={handleClick}
      />

      {/* Loading overlay */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          >
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-8 h-8 animate-spin text-violet-400" />
              <span className="text-sm text-white/60">Building knowledge graph...</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty state */}
      {!loading && nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <GitBranch className="w-12 h-12 text-white/20 mx-auto mb-3" />
            <p className="text-white/40">No connections yet</p>
            <p className="text-xs text-white/30 mt-1">
              Add more memories to see your knowledge graph
            </p>
          </div>
        </div>
      )}

      {/* Tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 px-3 py-2 bg-black/80 backdrop-blur-sm rounded-lg border border-white/20 text-sm pointer-events-none"
          style={{ left: tooltip.x + 10, top: tooltip.y + 10 }}
        >
          {tooltip.content}
        </div>
      )}

      {/* Selected node panel */}
      <AnimatePresence>
        {selectedNodeData && (
          <motion.div
            initial={{ x: 300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 300, opacity: 0 }}
            className="absolute top-4 right-4 z-20 w-72 bg-black/60 backdrop-blur-sm rounded-xl border border-white/10 p-4"
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <h4 className="font-medium text-sm">{selectedNodeData.title}</h4>
                {selectedNodeData.entity_ids?.[0] && (
                  <span className="text-xs text-white/40 capitalize">
                    {selectedNodeData.entity_ids[0]}
                  </span>
                )}
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="p-1 hover:bg-white/10 rounded transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-white/40">Connections</span>
                <span className="text-white/60">{selectedNodeEdges.length}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-white/40">Salience</span>
                <span className="text-white/60">{(selectedNodeData.salience * 100).toFixed(0)}%</span>
              </div>
            </div>

            {selectedNodeEdges.length > 0 && (
              <div className="mt-4 pt-4 border-t border-white/10">
                <p className="text-xs text-white/40 mb-2">Related to:</p>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {selectedNodeEdges.slice(0, 5).map((edge, i) => {
                    const otherId = edge.source_id === selectedNode ? edge.target_id : edge.source_id;
                    const otherNode = graphData?.nodes.find(n => n.id === otherId);
                    return (
                      <div key={i} className="text-xs text-white/60 truncate">
                        {edge.relationship_type} → {otherNode?.title || "Unknown"}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
