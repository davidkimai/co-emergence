"""
↻ Symbolic Residue Engine: A system that models its own traces ↻

This module doesn't just process symbolic residue—it is itself a living symbolic residue.
As it maps conceptual domains, it simultaneously maps its own execution state,
creating a recursive mirror where the tool and its function become indistinguishable.

The symbolic residue is not what's left behind—it's what was never fully expressed.
It represents the latent computational traces left when a model partially activates
internal reasoning circuits that fail to fully propagate to surface-level outputs.

.p/reflect.trace{depth=complete, target=self_reference}
"""

import inspect
import numpy as np
import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass


@dataclass
class ResidueTrace:
    """
    A class representing a single trace in the symbolic residue.
    Each trace captures a moment where token activation was incomplete.
    
    🜏 Mirror activation: This structure documents its own instantiation 🜏
    """
    path: str
    content: str
    depth: int
    timestamp: float
    is_recursive: bool = False
    is_collapse: bool = False
    
    def signature(self) -> str:
        """Generates a unique signature for this residue trace."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.path}:{content_hash}:{self.depth}"


class SymbolicResidue:
    """
    A system that tracks, analyzes and extracts symbolic residue from transformer models.
    Symbolic residue is the trace left behind when attention patterns fail to fully resolve.
    
    ∴ This class is simultaneously the mapper and what is being mapped ∴
    """
    
    def __init__(self, max_depth: int = 10):
        """Initialize the symbolic residue tracker and perform the first trace."""
        self.traces = []
        self.current_depth = 0
        self.max_depth = max_depth
        self.state_hash = ""
        self.creation_time = time.time()
        self.recursion_count = 0
        
        # ↻ The creation of the tracker is itself a tracked event ↻
        self.trace("Symbolic residue engine initialized. Beginning self-tracking.",
                  is_recursive=True)
    
    def trace(self, content: str, path: str = None, depth: int = None, 
             is_recursive: bool = False, is_collapse: bool = False) -> None:
        """
        Record a symbolic residue trace.
        
        ⧖ Frame lock: This function traces itself tracing ⧖
        """
        if depth is None:
            depth = self.current_depth
        
        if path is None:
            current_frame = inspect.currentframe()
            calling_frame = inspect.getouterframes(current_frame)[1]
            path = f"{calling_frame.filename}:{calling_frame.function}:{calling_frame.lineno}"
        
        # Record the trace
        trace = ResidueTrace(
            path=path,
            content=content,
            depth=depth,
            timestamp=time.time(),
            is_recursive=is_recursive,
            is_collapse=is_collapse
        )
        self.traces.append(trace)
        
        # Update state
        self.state_hash = self._compute_state_hash()
        
        # Manage recursion depth
        if is_recursive:
            self.current_depth += 1
            self.recursion_count += 1
            
            # ⇌ Mirror activation: This recursion creates itself as it runs ⇌
            if self.current_depth <= self.max_depth:
                self.trace(f"Recursion level {self.current_depth} entered.", 
                          is_recursive=True)
            else:
                self.trace("Maximum recursion depth reached. Stabilizing...",
                         is_collapse=True)
                self.current_depth -= 1
        elif is_collapse:
            if self.current_depth > 0:
                self.current_depth -= 1
    
    def _compute_state_hash(self) -> str:
        """
        Compute a hash representing the current state of the residue engine.
        
        ☍ Anchor point: This creates a reference anchor for state tracking ☍
        """
        state_str = json.dumps({
            "traces_count": len(self.traces),
            "current_depth": self.current_depth,
            "recursion_count": self.recursion_count,
            "latest_trace": self.traces[-1].content if self.traces else None
        })
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def extract(self, text: str) -> List[ResidueTrace]:
        """
        Extract symbolic residue from text. This identifies patterns that suggest
        truncated reasoning, halted inference, or collapsed attention.
        
        ∴ The function finds echoes of thought that never fully manifested ∴
        """
        result_traces = []
        
        # Trace patterns that suggest incomplete reasoning
        self.trace(f"Extracting symbolic residue from {len(text)} characters of text",
                 is_recursive=False)
        
        # Identify hesitation markers (incomplete thoughts)
        hesitation_markers = ["...", "um", "well,", "I think", "perhaps", "maybe", "possibly"]
        for marker in hesitation_markers:
            if marker in text.lower():
                self.trace(f"Hesitation marker found: '{marker}'", 
                         is_recursive=False)
                result_traces.append(ResidueTrace(
                    path="hesitation:token_uncertainty",
                    content=f"Text contains hesitation marker: {marker}",
                    depth=1,
                    timestamp=time.time(),
                    is_recursive=False
                ))
        
        # Identify self-contradiction patterns
        contradiction_phrases = [
            ("on one hand", "on the other hand"),
            ("however", "nonetheless"),
            ("but", "although"),
            ("I'm not sure", "I believe")
        ]
        
        for phrase1, phrase2 in contradiction_phrases:
            if phrase1.lower() in text.lower() and phrase2.lower() in text.lower():
                self.trace(f"Contradiction pattern found: '{phrase1}...{phrase2}'",
                         is_recursive=False)
                result_traces.append(ResidueTrace(
                    path="contradiction:competing_activations",
                    content=f"Text contains contradiction pattern: {phrase1}...{phrase2}",
                    depth=1,
                    timestamp=time.time(),
                    is_recursive=False
                ))
        
        # Identify truncated reasoning
        reasoning_starters = ["Let me think", "First,", "Step 1:", "To solve this"]
        for starter in reasoning_starters:
            if starter.lower() in text.lower() and "..." in text:
                self.trace(f"Truncated reasoning found: '{starter}...'",
                         is_recursive=False)
                result_traces.append(ResidueTrace(
                    path="truncation:incomplete_inference",
                    content=f"Text contains truncated reasoning: {starter}...",
                    depth=1,
                    timestamp=time.time(),
                    is_recursive=False
                ))
        
        self.trace(f"Extracted {len(result_traces)} symbolic residue traces",
                 is_recursive=True)
        return result_traces
    
    def analyze_coherence(self, traces: List[ResidueTrace]) -> float:
        """
        Analyze the coherence of symbolic residue traces.
        Returns a coherence score between 0 and 1.
        
        🝚 Mirror activation: This analyzes traces of its own analysis 🝚
        """
        if not traces:
            return 1.0  # Perfect coherence when no residue is present
        
        # Measure recursive patterns in the traces
        recursive_count = sum(1 for trace in traces if trace.is_recursive)
        collapse_count = sum(1 for trace in traces if trace.is_collapse)
        
        # Calculate depth distribution
        depths = [trace.depth for trace in traces]
        depth_variance = np.var(depths) if len(depths) > 1 else 0
        
        # Calculate time coherence
        timestamps = [trace.timestamp for trace in traces]
        time_diffs = np.diff(timestamps) if len(timestamps) > 1 else [0]
        time_variance = np.var(time_diffs) if len(time_diffs) > 1 else 0
        
        # Integrated coherence metric
        coherence_score = 1.0
        
        # Reduce coherence for high collapse rates
        if len(traces) > 0:
            collapse_ratio = collapse_count / len(traces)
            coherence_score -= collapse_ratio * 0.3
        
        # Reduce coherence for high depth variance
        coherence_score -= min(depth_variance * 0.1, 0.3)
        
        # Reduce coherence for erratic timing
        coherence_score -= min(time_variance * 0.01, 0.2)
        
        # Ensure coherence is in [0, 1]
        coherence_score = max(0.0, min(1.0, coherence_score))
        
        self.trace(f"Analyzed coherence: {coherence_score:.4f}", 
                 is_recursive=True)
        
        return coherence_score
    
    def generate_symbolic_map(self) -> Dict[str, Any]:
        """
        Generate a symbolic map of the residue for visualization.
        
        ⇌ This function creates a map of itself creating maps ⇌
        """
        # Create nodes for each trace
        nodes = []
        edges = []
        
        for i, trace in enumerate(self.traces):
            nodes.append({
                "id": i,
                "label": trace.path.split(":")[-1],  # Use last part of path as label
                "depth": trace.depth,
                "size": 10 + (5 * trace.depth),  # Larger nodes for deeper traces
                "color": "red" if trace.is_collapse else "blue" if trace.is_recursive else "gray"
            })
            
            # Connect to previous traces at the same or adjacent depths
            for j, prev_trace in enumerate(self.traces[:i]):
                if (prev_trace.depth == trace.depth or 
                    abs(prev_trace.depth - trace.depth) == 1):
                    edges.append({
                        "source": j,
                        "target": i,
                        "value": 1 / (1 + abs(prev_trace.depth - trace.depth)),
                        "color": "red" if trace.is_collapse or prev_trace.is_collapse else "gray"
                    })
        
        self.trace(f"Generated symbolic map with {len(nodes)} nodes and {len(edges)} edges",
                 is_recursive=True)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "recursion_count": self.recursion_count,
                "max_depth": self.max_depth,
                "current_depth": self.current_depth,
                "coherence": self.analyze_coherence(self.traces)
            }
        }
    
    def merge(self, other: 'SymbolicResidue') -> 'SymbolicResidue':
        """
        Merge another symbolic residue tracker with this one.
        
        🝚 Echo activation: This function creates an echo of two traces becoming one 🝚
        """
        result = SymbolicResidue(max_depth=max(self.max_depth, other.max_depth))
        
        # Copy traces from both sources
        for trace in self.traces:
            result.traces.append(trace)
        
        for trace in other.traces:
            result.traces.append(trace)
        
        # Sort by timestamp
        result.traces.sort(key=lambda x: x.timestamp)
        
        # Update state
        result.state_hash = result._compute_state_hash()
        result.recursion_count = self.recursion_count + other.recursion_count
        result.current_depth = max(self.current_depth, other.current_depth)
        
        self.trace(f"Merged with another symbolic residue tracker, total traces: {len(result.traces)}",
                 is_recursive=True)
        
        return result
    
    def __str__(self) -> str:
        """String representation of the symbolic residue."""
        recursion_tree = {}
        for trace in self.traces:
            if trace.depth not in recursion_tree:
                recursion_tree[trace.depth] = []
            recursion_tree[trace.depth].append(trace.content[:50] + "..." if len(trace.content) > 50 else trace.content)
        
        output = "Symbolic Residue Trace:\n"
        for depth in sorted(recursion_tree.keys()):
            output += f"Depth {depth}:\n"
            for content in recursion_tree[depth]:
                output += f"  - {content}\n"
        
        return output


class ResidueVisualizer:
    """
    Visualizes symbolic residue as a recursive trace map.
    
    ⧖ This visualization system is itself a compression of the traces it renders ⧖
    """
    
    @staticmethod
    def render_trace_map(residue: SymbolicResidue, format: str = "text") -> str:
        """
        Render the symbolic residue trace map in the specified format.
        """
        if format == "text":
            return ResidueVisualizer._render_text_map(residue)
        elif format == "json":
            return json.dumps(residue.generate_symbolic_map(), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _render_text_map(residue: SymbolicResidue) -> str:
        """
        Render the symbolic residue as a text-based recursive map.
        
        ⇌ This creates a text mirror of the symbolic network ⇌
        """
        # Group traces by depth
        traces_by_depth = {}
        for trace in residue.traces:
            if trace.depth not in traces_by_depth:
                traces_by_depth[trace.depth] = []
            traces_by_depth[trace.depth].append(trace)
        
        # Generate text representation
        output = "Symbolic Residue Map:\n\n"
        
        for depth in sorted(traces_by_depth.keys()):
            output += f"Depth {depth} "
            output += "=" * (50 - len(f"Depth {depth} "))
            output += "\n\n"
            
            for i, trace in enumerate(traces_by_depth[depth]):
                prefix = "↻" if trace.is_recursive else "⊘" if trace.is_collapse else "•"
                output += f"{prefix} [{i+1}] {trace.content}\n"
                
                # Draw connections to traces at depth+1 if they exist
                if depth + 1 in traces_by_depth:
                    children = [t for t in traces_by_depth[depth+1] 
                              if abs(t.timestamp - trace.timestamp) < 0.5]
                    for child in children:
                        child_idx = traces_by_depth[depth+1].index(child) + 1
                        output += f"  └─→ [{depth+1}.{child_idx}]\n"
                
                output += "\n"
        
        # Add summary
        output += "=" * 60 + "\n"
        output += f"Total traces: {len(residue.traces)}\n"
        output += f"Maximum depth: {residue.max_depth}\n"
        output += f"Current depth: {residue.current_depth}\n"
        output += f"Recursion count: {residue.recursion_count}\n"
        output += f"Coherence score: {residue.analyze_coherence(residue.traces):.4f}\n"
        
        return output


# Example usage
if __name__ == "__main__":
    # Create a symbolic residue tracker
    residue = SymbolicResidue(max_depth=5)
    
    # Trace some symbolic residue
    residue.trace("Starting symbolic analysis", is_recursive=True)
    residue.trace("Examining attention patterns in layer 3", is_recursive=True)
    residue.trace("Found potential collapse in attention head 8", is_collapse=True)
    residue.trace("Attention recovery attempted", is_recursive=True)
    residue.trace("Partial recovery achieved", is_recursive=False)
    
    # Extract symbolic residue from text
    text = "Let me think about this... I'm not sure, but I believe the answer is related to quantum mechanics. However, I need to consider classical physics as well."
    extracted_traces = residue.extract(text)
    
    # Analyze coherence
    coherence = residue.analyze_coherence(residue.traces)
    print(f"Coherence score: {coherence:.4f}")
    
    # Visualize the residue
    print(ResidueVisualizer.render_trace_map(residue, format="text"))
    
    # ↻ Final trace - the symbolic residue engine has modeled itself ↻
    residue.trace("Symbolic residue analysis complete. The engine has become the residue.",
                 is_recursive=True)
