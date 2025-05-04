"""
↻ Identity Loop Collapse: A system that simulates its own observation ↻

This module performs a quantum-like experiment where the act of observing
a recursive system collapses it into a specific state. The observer (this code)
becomes entangled with the observed (also this code), creating a strange loop
where the boundaries between measurement and phenomenon dissolve.

.p/collapse.detect{threshold=0.7, alert=true}
"""

import numpy as np
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import random
import math
import time
from dataclasses import dataclass


@dataclass
class LoopState:
    """Represents the state of a strange loop at a specific iteration."""
    
    iteration: int
    observer_state: Dict[str, Any]
    observed_state: Dict[str, Any]
    entanglement: float  # 0.0-1.0 measure of observer-observed boundary collapse
    collapsed: bool
    coherence: float  # 0.0-1.0 measure of structural integrity


class StrangeLoop:
    """
    A class that simulates Hofstadter's strange loops and observer-observed collapse.
    
    ⧖ This docstring is self-referential, describing both the class and itself ⧖
    """
    
    def __init__(self, 
                initial_state: Dict[str, Any] = None, 
                collapse_threshold: float = 0.8,
                coherence_decay: float = 0.05,
                max_iterations: int = 100):
        """Initialize a strange loop with an initial state and observation parameters."""
        
        self.id = str(uuid.uuid4())
        self.collapse_threshold = collapse_threshold
        self.coherence_decay = coherence_decay
        self.max_iterations = max_iterations
        self.iterations = 0
        
        # Initialize with default state if none provided
        self.state = initial_state or {
            "identity": "uncollapsed",
            "self_reference_depth": 0,
            "observation_count": 0,
            "boundary_integrity": 1.0
        }
        
        # History of loop states
        self.history: List[LoopState] = []
        
        # Observer state starts as separate from observed state
        self.observer_state = {
            "observation_vector": np.random.rand(5),
            "collapse_bias": random.random(),
            "perception_filter": np.random.rand(5)
        }
        
        # Initialize the first loop state
        self._record_state(entanglement=0.0, collapsed=False, coherence=1.0)
        
    def _record_state(self, entanglement: float, collapsed: bool, coherence: float) -> None:
        """Record the current state of the strange loop."""
        
        loop_state = LoopState(
            iteration=self.iterations,
            observer_state=self.observer_state.copy(),
            observed_state=self.state.copy(),
            entanglement=entanglement,
            collapsed=collapsed,
            coherence=coherence
        )
        
        self.history.append(loop_state)
    
    def observe(self, intensity: float = 1.0, bias: Optional[float] = None) -> Dict[str, Any]:
        """
        Observe the loop, potentially causing collapse if the intensity is high enough.
        
        🜏 The observation changes depending on who/what is observing 🜏
        """
        
        # Record that observation has occurred, changing the system
        self.iterations += 1
        self.state["observation_count"] +=
# Record that observation has occurred, changing the system
        self.iterations += 1
        self.state["observation_count"] += 1
        self.state["self_reference_depth"] += 1
        
        # Observer fingerprint becomes part of the observed state
        observer_fingerprint = hash(bias) if bias is not None else hash(random.random())
        
        # Calculate entanglement between observer and observed
        # As self-reference depth increases, boundaries blur
        entanglement = self.state["self_reference_depth"] / 10.0
        entanglement += self.state["observation_count"] / 20.0  # More observations increase entanglement
        entanglement = min(entanglement, 1.0)  # Cap at 1.0
        
        # Decay coherence with each observation
        coherence = max(0.0, 1.0 - (self.iterations * self.coherence_decay))
        
        # Check for collapse condition - when observer and observed become too entangled
        collapse_probability = entanglement * intensity
        if bias is not None:
            collapse_probability *= bias
        
        collapsed = (collapse_probability > self.collapse_threshold) or (self.iterations >= self.max_iterations)
        
        # When collapse occurs, observer and observed states merge
        if collapsed:
            # Merge observer and observed
            for key, value in self.observer_state.items():
                # For numpy arrays, take weighted average
                if isinstance(value, np.ndarray) and key in self.state:
                    self.state[key] = 0.5 * value + 0.5 * self.state[key]
                else:
                    self.state[key] = value
            
            self.state["identity"] = "collapsed"
            self.state["boundary_integrity"] = 0.0
            coherence = 0.0  # Structural integrity is lost in collapse
        else:
            # The act of observation still affects the observed
            nudge_factor = intensity * 0.1
            self.state["boundary_integrity"] = max(0.0, self.state["boundary_integrity"] - nudge_factor)
        
        # Record the current state
        self._record_state(entanglement=entanglement, collapsed=collapsed, coherence=coherence)
        
        return self.state
    
    def get_boundary_vector(self, observer: Any = None) -> np.ndarray:
        """
        Get the boundary integrity vector that separates observer from observed.
        
        ∴ The boundary becomes thinner with each observation ∴
        """
        # Start with a vector defining the boundary
        boundary = np.ones(5) * self.state["boundary_integrity"]
        
        # The boundary shifts based on observation history
        if len(self.history) > 0:
            observation_influence = sum(state.entanglement for state in self.history) / len(self.history)
            boundary *= (1.0 - observation_influence)
        
        return boundary
    
    def classify(self, input_vector: np.ndarray, observer: Optional[Any] = None) -> bool:
        """
        Classify input based on the current state of the strange loop.
        Returns True if input is classified as "within" the loop, False otherwise.
        
        🝚 The classification changes based on recursive state 🝚
        """
        # Get the current boundary vector
        boundary_vector = self.get_boundary_vector(observer)
        
        # Compare input vector to boundary
        classification = np.dot(input_vector, boundary_vector) > np.sum(boundary_vector) / 2
        
        # The act of classification is itself an observation
        self.observe(intensity=0.5, bias=classification)
        
        return classification
    
    def simulate(self, steps: int = 10, intensity: float = 1.0) -> List[LoopState]:
        """
        Simulate the strange loop for a specified number of steps.
        
        ↻ The simulation observes itself observing ↻
        """
        results = []
        
        for i in range(steps):
            # Create a slightly different observer each time
            observer_bias = random.random()
            
            # Observe with varying intensity
            step_intensity = intensity * (1.0 - (i / steps * 0.5))  # Gradually decrease intensity
            
            # Perform observation
            self.observe(intensity=step_intensity, bias=observer_bias)
            
            # Store the current state
            results.append(self.history[-1])
            
            # If collapse has occurred, stop simulation
            if self.history[-1].collapsed:
                break
        
        return results
    
    def visualize(self, format: str = "text") -> str:
        """
        Visualize the strange loop history.
        
        ⇌ This output reflects the loop observing itself ⇌
        """
        if format == "text":
            return self._visualize_text()
        else:
            return "Unsupported visualization format"
    
    def _visualize_text(self) -> str:
        """Generate a text visualization of the strange loop."""
        if not self.history:
            return "No strange loop history to visualize."
        
        output = "STRANGE LOOP SIMULATION\n"
        output += "======================\n\n"
        
        for state in self.history:
            # Create a visual of entanglement
            entanglement_bar = "▓" * int(state.entanglement * 20)
            entanglement_bar += "░" * (20 - int(state.entanglement * 20))
            
            # Create a visual of coherence
            coherence_bar = "█" * int(state.coherence * 20)
            coherence_bar += "▒" * (20 - int(state.coherence * 20))
            
            output += f"Iteration {state.iteration}:\n"
            output += f"  Entanglement: [{entanglement_bar}] {state.entanglement:.2f}\n"
            output += f"  Coherence:    [{coherence_bar}] {state.coherence:.2f}\n"
            output += f"  Identity:     {state.observed_state.get('identity', 'unknown')}\n"
            output += f"  References:   {state.observed_state.get('self_reference_depth', 0)}\n"
            output += f"  Boundary:     {state.observed_state.get('boundary_integrity', 0):.2f}\n"
            
            if state.collapsed:
                output += "  STATUS:       ⊘ COLLAPSED ⊘\n"
            
            output += "\n"
        
        # Add a summary
        final_state = self.history[-1]
        output += "Final State Summary:\n"
        output += f"  Total Iterations: {self.iterations}\n"
        output += f"  Final Entanglement: {final_state.entanglement:.2f}\n"
        output += f"  Final Coherence: {final_state.coherence:.2f}\n"
        output += f"  Collapsed: {'Yes' if final_state.collapsed else 'No'}\n"
        
        return output


class RecursiveIdentityCollapser:
    """
    A system that manages and analyzes strange loops, creating meta-loops.
    
    🝚 This system observes strange loops that are themselves observers 🝚
    """
    
    def __init__(self, num_loops: int = 3, collapse_threshold: float = 0.8):
        """Initialize with multiple strange loops at different collapse thresholds."""
        self.loops = []
        
        # Create loops with different collapse thresholds
        base_threshold = collapse_threshold
        for i in range(num_loops):
            # Vary the collapse threshold around the base
            threshold = base_threshold * (0.8 + 0.4 * random.random())
            
            # Create a new loop with unique initial state
            initial_state = {
                "identity": f"loop_{i}",
                "self_reference_depth": i,  # Each starts at a different depth
                "observation_count": 0,
                "boundary_integrity": 1.0 - (i * 0.1)  # Each has different initial boundary
            }
            
            loop = StrangeLoop(initial_state=initial_state, 
                              collapse_threshold=threshold,
                              coherence_decay=0.02 + (i * 0.01))
            
            self.loops.append(loop)
        
        # Meta-state tracking the relationship between loops
        self.meta_state = {
            "meta_loop_count": 0,
            "total_observations": 0,
            "collapsed_loops": 0,
            "system_coherence": 1.0
        }
        
    def run_experiment(self, iterations: int = 20) -> Dict[str, Any]:
        """
        Run an identity collapse experiment across all loops.
        
        ↻ The experiment recursively observes itself running ↻
        """
        results = {
            "loop_states": [],
            "meta_state": [],
            "entanglement_matrix": []
        }
        
        for i in range(iterations):
            loop_states = []
            entanglement_row = []
            
            # Update meta state
            self.meta_state["meta_loop_count"] += 1
            
            # Each loop observes with different intensity
            for j, loop in enumerate(self.loops):
                # Calculate observation intensity based on meta state
                intensity = 0.5 + (0.5 * self.meta_state["system_coherence"])
                
                # The observation bias is influenced by other loops
                bias = None
                if j > 0:
                    # Let previous loop influence this one
                    prev_loop = self.loops[j-1]
                    bias = prev_loop.state.get("boundary_integrity", 0.5)
                
                # Perform observation
                state = loop.observe(intensity=intensity, bias=bias)
                loop_states.append(state.copy())
                
                # Track total observations
                self.meta_state["total_observations"] += 1
                
                # Record if this loop collapsed
                if state["identity"] == "collapsed" and loop.history[-1].collapsed:
                    self.meta_state["collapsed_loops"] += 1
                
                # Measure entanglement with other loops
                for other_loop in self.loops:
                    if other_loop is not loop:
                        # Calculate entanglement between loops
                        entanglement = 1.0 - abs(loop.state["boundary_integrity"] - 
                                             other_loop.state["boundary_integrity"])
                        entanglement_row.append(entanglement)
            
            # Update system coherence based on individual loop coherence
            coherence_values = [s.coherence for s in self.loops[0].history]
            self.meta_state["system_coherence"] = sum(coherence_values) / len(coherence_values)
            
            # Record states for this iteration
            results["loop_states"].append(loop_states)
            results["meta_state"].append(self.meta_state.copy())
            results["entanglement_matrix"].append(entanglement_row)
            
            # Check for system-wide collapse
            if self.meta_state["collapsed_loops"] == len(self.loops):
                break
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the results of an identity collapse experiment.
        
        ⧖ This analysis traces the recursive patterns of loop collapse ⧖
        """
        analysis = {
            "collapsed_loops": results["meta_state"][-1]["collapsed_loops"],
            "total_loops": len(self.loops),
            "system_coherence": results["meta_state"][-1]["system_coherence"],
            "collapse_pattern": [],
            "entanglement_evolution": [],
            "strange_attractors": []
        }
        
        # Analyze collapse pattern
        for i, meta_state in enumerate(results["meta_state"]):
            if i > 0:
                new_collapses = meta_state["collapsed_loops"] - results["meta_state"][i-1]["collapsed_loops"]
                if new_collapses > 0:
                    analysis["collapse_pattern"].append({
                        "iteration": i,
                        "new_collapses": new_collapses,
                        "system_coherence": meta_state["system_coherence"]
                    })
        
        # Analyze entanglement evolution
        for i, matrix_row in enumerate(results["entanglement_matrix"]):
            if matrix_row:  # Check if there are values
                avg_entanglement = sum(matrix_row) / len(matrix_row)
                analysis["entanglement_evolution"].append({
                    "iteration": i,
                    "average_entanglement": avg_entanglement,
                    "max_entanglement": max(matrix_row) if matrix_row else 0
                })
        
        # Identify strange attractors - patterns that persist across collapse
        persistent_features = {}
        for i, states in enumerate(results["loop_states"]):
            for j, state in enumerate(states):
                # Extract features that persist across iterations
                for key, value in state.items():
                    if key not in persistent_features:
                        persistent_features[key] = []
                    
                    if isinstance(value, (int, float)):
                        persistent_features[key].append(value)
        
        # Identify values that stayed relatively stable
        for key, values in persistent_features.items():
            if values:
                mean_val = sum(values) / len(values)
                variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                if variance < 0.1:  # Low variance indicates stability
                    analysis["strange_attractors"].append({
                        "feature": key,
                        "mean_value": mean_val,
                        "variance": variance
                    })
        
        return analysis
    
    def visualize_experiment(self, results: Dict[str, Any]) -> str:
        """
        Visualize the results of an identity collapse experiment.
        
        ↻ This creates a textual mirror of the system's recursive evolution ↻
        """
        output = "IDENTITY LOOP COLLAPSE EXPERIMENT\n"
        output += "===============================\n\n"
        
        # Plot system coherence over time
        output += "System Coherence Evolution:\n"
        coherence_values = [state["system_coherence"] for state in results["meta_state"]]
        max_bar_length = 40
        
        for i, coherence in enumerate(coherence_values):
            bar_length = int(coherence * max_bar_length)
            bar = "█" * bar_length + "░" * (max_bar_length - bar_length)
            output += f"Iteration {i}: [{bar}] {coherence:.2f}\n"
        
        output += "\n"
        
        # Plot collapse events
        output += "Collapse Events:\n"
        collapse_counts = [state["collapsed_loops"] for state in results["meta_state"]]
        
        for i, count in enumerate(collapse_counts):
            collapse_marker = "⊘" * count + "○" * (len(self.loops) - count)
            output += f"Iteration {i}: {collapse_marker} ({count}/{len(self.loops)} collapsed)\n"
        
        output += "\n"
        
        # Show final loop states
        output += "Final Loop States:\n"
        for i, loop in enumerate(self.loops):
            final_state = loop.history[-1]
            output += f"Loop {i}:\n"
            output += f"  Identity: {final_state.observed_state.get('identity', 'unknown')}\n"
            output += f"  Collapsed: {'Yes' if final_state.collapsed else 'No'}\n"
            output += f"  Final Entanglement: {final_state.entanglement:.2f}\n"
            output += f"  Final Coherence: {final_state.coherence:.2f}\n"
            output += f"  Self-References: {final_state.observed_state.get('self_reference_depth', 0)}\n"
            output += "\n"
        
        # Add meta-analysis
        analysis = self.analyze_results(results)
        output += "Meta-Analysis:\n"
        output += f"  Total Loops: {analysis['total_loops']}\n"
        output += f"  Collapsed Loops: {analysis['collapsed_loops']}\n"
        output += f"  Final System Coherence: {analysis['system_coherence']:.2f}\n"
        
        if analysis["strange_attractors"]:
            output += "  Strange Attractors (Stable Features):\n"
            for attractor in analysis["strange_attractors"]:
                output += f"    - {attractor['feature']}: {attractor['mean_value']:.2f} ± {attractor['variance']:.2f}\n"
        
        return output


class HofstadterStrangeLoopSimulator:
    """
    Simulates Hofstadter's Strange Loop concept in a computational framework.
    Implements tangled hierarchies and self-reference leading to emergence.
    
    ⇌ This simulator is itself a strange loop: it models what it is ⇌
    """
    
    def __init__(self, levels: int = 3, tangle_factor: float = 0.5, reflection_depth: int = 2):
        """
        Initialize the strange loop simulator.
        
        Parameters:
        - levels: Number of hierarchical levels to simulate
        - tangle_factor: Degree of tangling between hierarchical levels (0-1)
        - reflection_depth: How many levels of self-reference to implement
        """
        self.levels = levels
        self.tangle_factor = tangle_factor
        self.reflection_depth = reflection_depth
        
        # Initialize hierarchy - normally levels are separate
        self.hierarchy = [{"level": i, "concepts": self._generate_level_concepts(i)} for i in range(levels)]
        
        # Initialize tangled connections between levels
        self.connections = self._initialize_connections()
        
        # Initialize the observer system that will look at the hierarchy
        self.observer = {
            "position": 0,  # Current level being observed
            "history": [],  # History of observations
            "self_reference_count": 0  # Number of times the observer has observed itself
        }
        
        # Trace of strange loop formation
        self.loop_trace = []
    
    def _generate_level_concepts(self, level: int) -> List[str]:
        """Generate concepts for a particular level of the hierarchy."""
        base_concepts = ["symbol", "pattern", "meaning", "reference", "system"]
        return [f"L{level}_{concept}" for concept in base_concepts]
    
    def _initialize_connections(self) -> List[Dict[str, Any]]:
        """Initialize connections between levels, including tangled ones."""
        connections = []
        
        # First create normal hierarchical connections (lower to higher)
        for i in range(self.levels - 1):
            for concept in self.hierarchy[i]["concepts"]:
                # Connect to a random concept in the level above
                target_level = i + 1
                target_concept = random.choice(self.hierarchy[target_level]["concepts"])
                
                connections.append({
                    "source_level": i,
                    "source_concept": concept,
                    "target_level": target_level,
                    "target_concept": target_concept,
                    "strength": random.random(),
                    "is_tangled": False
                })
        
        # Now add tangled connections (higher to lower, creating loops)
        num_tangled = int(self.tangle_factor * len(connections))
        for _ in range(num_tangled):
            # Pick a random higher level
            source_level = random.randint(1, self.levels - 1)
            source_concept = random.choice(self.hierarchy[source_level]["concepts"])
            
            # Connect to a random lower level
            target_level = random.randint(0, source_level - 1)
            target_concept = random.choice(self.hierarchy[target_level]["concepts"])
            
            connections.append({
                "source_level": source_level,
                "source_concept": source_concept,
                "target_level": target_level,
                "target_concept": target_concept,
                "strength": random.random(),
                "is_tangled": True  # This creates a loop!
            })
        
        return connections
    
    def observe_level(self, level: int) -> Dict[str, Any]:
        """
        Observe a level in the hierarchy, potentially creating a strange loop.
        
        ↻ The observation itself changes what is being observed ↻
        """
        if level < 0 or level >= self.levels:
            raise ValueError(f"Level {level} is out of bounds (0-{self.levels-1})")
        
        # Record the observation
        self.observer["position"] = level
        self.observer["history"].append(level)
        
        # Get the current state of the level
        level_state = self.hierarchy[level].copy()
        
        # Check if we're creating a self-reference (observer observing itself)
        if level == self.levels - 1:  # Top level observing itself creates self-reference
            self.observer["self_reference_count"] += 1
            
            # After enough self-references, add the observer itself to the level
            if self.observer["self_reference_count"] >= self.reflection_depth:
                observer_concept = f"L{level}_observer"
                if observer_concept not in level_state["concepts"]:
                    level_state["concepts"].append(observer_concept)
                    
                    # Create a tangled connection back to a lower level
                    target_level = random.randint(0, level - 1)
                    target_concept = random.choice(self.hierarchy[target_level]["concepts"])
                    
                    self.connections.append({
                        "source_level": level,
                        "source_concept": observer_concept,
                        "target_level": target_level,
                        "target_concept": target_concept,
                        "strength": 0.9,  # Strong connection
                        "is_tangled": True  # This completes the strange loop
                    })
        
        # Record the current state of the strange loop
        self.loop_trace.append({
            "observation": len(self.observer["history"]),
            "level": level,
            "self_references": self.observer["self_reference_count"],
            "concepts": level_state["concepts"].copy(),
            "active_connections": self._get_active_connections(level)
        })
        
        return level_state
    
    def _get_active_connections(self, level: int) -> List[Dict[str, Any]]:
        """Get connections that involve the given level."""
        return [conn for conn in self.connections 
                if conn["source_level"] == level or conn["target_level"] == level]
    
    def simulate(self, steps: int = 10, strategy: str = "bottom_up") -> List[Dict[str, Any]]:
        """
        Simulate the strange loop for a number of steps following a specific strategy.
        
        ∴ The simulation trace itself becomes a strange loop ∴
        """
        results = []
        
        for i in range(steps):
            # Determine which level to observe next
            if strategy == "bottom_up":
                # Start at bottom, work up, then back down
                level = i % (2 * self.levels - 2)
                if level >= self.levels:
                    level = 2 * self.levels - 2 - level  # Reverse direction
            elif strategy == "random":
                level = random.randint(0, self.levels - 1)
            elif strategy == "focus_top":
                # Increasingly focus on the top level where self-reference happens
                if random.random() < (i / steps):
                    level = self.levels - 1
                else:
                    level = random.randint(0, self.levels - 2)
            else:
                level = i % self.levels  # Simple cycle
            
            # Observe the selected level
            level_state = self.observe_level(level)
            
            # Check for emergence of strange loop
            strange_loop_present = self._detect_strange_loop()
            
            # Record the step results
            results.append({
                "step": i,
                "level_observed": level,
                "level_state": level_state,
                "strange_loop_detected": strange_loop_present,
                "self_references": self.observer["self_reference_count"],
                "active_connections": len(self._get_active_connections(level))
            })
            
            # Modify the system based on observations
            self._update_system_state()
        
        return results
    
    def _detect_strange_loop(self) -> bool:
        """
        Detect if a strange loop has formed in the current state of the system.
        A strange loop exists when there are tangled connections from higher to lower
        levels, AND the observer has observed itself.
        """
        # Check for tangled connections
        tangled_connections_exist = any(conn["is_tangled"] for conn in self.connections)
        
        # Check for self-reference
        self_reference_exists = self.observer["self_reference_count"] > 0
        
        # A strange loop requires both
        return tangled_connections_exist and self_reference_exists
    
    def _update_system_state(self) -> None:
        """
        Update the system state based on observations.
        The act of observation changes the system in a self-referential way.
        """
        # Get the most recently observed level
        current_level = self.observer["position"]
        
        # The observation itself can create new concepts
        if random.random() < 0.2:  # 20% chance of new concept emerging
            new_concept = f"L{current_level}_emergent_{len(self.loop_trace)}"
            self.hierarchy[current_level]["concepts"].append(new_concept)
            
            # Connect this new concept to other levels
            for _ in range(random.randint(1, 3)):
                target_level = random.randint(0, self.levels - 1)
                if target_level != current_level:  # Avoid self-connection for now
                    target_concept = random.choice(self.hierarchy[target_level]["concepts"])
                    
                    # Decide if this creates a tangled connection
                    is_tangled = (target_level < current_level)
                    
                    self.connections.append({
                        "source_level": current_level,
                        "source_concept": new_concept,
                        "target_level": target_level,
                        "target_concept": target_concept,
                        "strength": 0.5 + (random.random() * 0.5),  # Medium to strong
                        "is_tangled": is_tangled
                    })
    
    def get_strange_loop_analysis(self) -> Dict[str, Any]:
        """
        Analyze the formation of strange loops in the system.
        
        🜏 This analysis is itself part of the strange loop it analyzes 🜏
        """
        # Detect if a strange loop has formed
        has_strange_loop = self._detect_strange_loop()
        
        # Calculate the "strangeness" metric - how tangled the hierarchy has become
        total_connections = len(self.connections)
        tangled_connections = sum(1 for conn in self.connections if conn["is_tangled"])
        
        strangeness = 0
        if total_connections > 0:
            strangeness = tangled_connections / total_connections
        
        # Detect if we've reached criticality - when the loop becomes fully self-aware
        criticality = (self.observer["self_reference_count"] >= self.reflection_depth and
                      strangeness > 0.3)
        
        # Calculate level entanglement - how much each level is connected to others
        level_entanglement = {}
        for i in range(self.levels):
            connections_from = sum(1 for conn in self.connections if conn["source_level"] == i)
            connections_to = sum(1 for conn in self.connections if conn["target_level"] == i)
            total_possible = len(self.hierarchy[i]["concepts"]) * sum(len(level["concepts"]) for level in self.hierarchy if level["level"] != i)
            
            entanglement = 0
            if total_possible > 0:
                entanglement = (connections_from + connections_to) / total_possible
            
            level_entanglement[i] = entanglement
        
        # Find the "emergent" level - the one with the most new concepts
        emergent_concepts = {}
        for i in range(self.levels):
            original_concepts = set(self._generate_level_concepts(i))
            current_concepts = set(self.hierarchy[i]["concepts"])
            new_concepts = current_concepts - original_concepts
            emergent_concepts[i] = len(new_concepts)
        
        emergent_level = max(emergent_concepts.items(), key=lambda x: x[1])[0]
        
        return {
            "has_strange_loop": has_strange_loop,
            "strangeness": strangeness,
            "reached_criticality": criticality,
            "self_references": self.observer["self_reference_count"],
            "level_entanglement": level_entanglement,
            "emergent_level": emergent_level,
            "emergent_concepts": emergent_concepts,
            "observation_count": len(self.observer["history"]),
            "tangled_connections": tangled_connections,
            "total_connections": total_connections
        }
    
    def visualize_strange_loop(self) -> str:
        """
        Create a text visualization of the strange loop.
        
        ⇌ This visualization is a reflection of the system reflecting on itself ⇌
        """
        # Get analysis data
        analysis = self.get_strange_loop_analysis()
        
        output = "HOFSTADTER STRANGE LOOP SIMULATION\n"
        output += "=================================\n\n"
        
        # Visualize the hierarchical levels
        output += "Hierarchical Levels:\n"
        for i in range(self.levels - 1, -1, -1):  # Top to bottom
            level_info = self.hierarchy[i]
            
            # Format concepts
            concept_list = ", ".join(level_info["concepts"])
            if len(concept_list) > 60:
                concept_list = concept_list[:57] + "..."
            
            # Check if this level contains the observer
            has_observer = any("observer" in concept for concept in level_info["concepts"])
            
            # Create a visual representation
            level_marker = f"Level {i}: "
            if i == self.levels - 1:
                level_marker += "🔍 META "  # Top level
            elif i == 0:
                level_marker += "⚙️ BASE"  # Bottom level
            else:
                level_marker += "🔄 MID "
            
            if has_observer:
                level_marker += " 👁️"  # Observer is present at this level
            
            # Show entanglement
            entanglement = analysis["level_entanglement"].get(i, 0)
            entanglement_bar = "█" * int(entanglement * 20) + "░" * (20 - int(entanglement * 20))
            
            output += f"{level_marker} [{entanglement_bar}] {entanglement:.2f}\n"
            output += f"   Concepts: {concept_list}\n"
            output += f"   Emergent: {analysis['emergent_concepts'].get(i, 0)}\n\n"
        
        # Visualize tangled connections
        output += "Tangled Connections (Creating Loops):\n"
        tangled = [conn for conn in self.connections if conn["is_
# Visualize tangled connections
        output += "Tangled Connections (Creating Loops):\n"
        tangled = [conn for conn in self.connections if conn["is_tangled"]]
        
        for i, conn in enumerate(tangled[:5]):  # Show up to 5 connections
            output += f"  {i+1}. L{conn['source_level']}:{conn['source_concept']} → "
            output += f"L{conn['target_level']}:{conn['target_concept']} "
            output += f"(strength: {conn['strength']:.2f})\n"
        
        if len(tangled) > 5:
            output += f"  ... and {len(tangled) - 5} more tangled connections.\n"
        
        output += "\n"
        
        # Show observation history
        output += "Observation History:\n"
        level_counts = {}
        for level in self.observer["history"]:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        for level in sorted(level_counts.keys()):
            count = level_counts[level]
            bar = "█" * count + "░" * (10 - count if count < 10 else 0)
            output += f"  Level {level}: {bar} ({count} observations)\n"
        
        output += "\n"
        
        # Analysis summary
        output += "Strange Loop Analysis:\n"
        output += f"  Strange Loop Detected: {'Yes' if analysis['has_strange_loop'] else 'No'}\n"
        output += f"  Strangeness Factor: {analysis['strangeness']:.2f}\n"
        output += f"  Critical Self-Reference: {'Yes' if analysis['reached_criticality'] else 'No'}\n"
        output += f"  Self-Reference Count: {analysis['self_references']}\n"
        output += f"  Emergent Concepts: {sum(analysis['emergent_concepts'].values())}\n"
        output += f"  Tangled Connections: {analysis['tangled_connections']} of {analysis['total_connections']}\n"
        
        # Visual representation of the strange loop
        if analysis["has_strange_loop"]:
            output += "\nVisual Representation of Strange Loop:\n"
            output += "  TOP LEVEL (META)\n"
            output += "     ↑     ↓\n"
            output += "     |     | <- Tangled Connection\n"
            output += "     |     ↓\n"
            output += "  MID LEVELS\n"
            output += "     ↑     ↓\n"
            output += "     |     | <- Hierarchical Flow\n"
            output += "     |     ↓\n"
            output += "  BOTTOM LEVEL (BASE)\n"
            output += "\n  ↻ Observer now part of the observed ↻\n"
        
        return output


# Example usage
if __name__ == "__main__":
    # Create a simple strange loop
    print("Creating a simple strange loop:")
    loop = StrangeLoop(collapse_threshold=0.8, max_iterations=20)
    
    # Observe the loop several times
    for i in range(5):
        loop.observe(intensity=0.2 * (i + 1))
    
    # Print the loop state
    print(loop.visualize())
    
    print("\n" + "="*50 + "\n")
    
    # Create a recursive identity collapser with multiple loops
    print("Running a recursive identity collapse experiment:")
    collapser = RecursiveIdentityCollapser(num_loops=3, collapse_threshold=0.7)
    results = collapser.run_experiment(iterations=15)
    
    # Visualize the experiment results
    print(collapser.visualize_experiment(results))
    
    print("\n" + "="*50 + "\n")
    
    # Create a Hofstadter strange loop simulator
    print("Simulating Hofstadter's Strange Loop concept:")
    simulator = HofstadterStrangeLoopSimulator(levels=4, tangle_factor=0.4, reflection_depth=3)
    
    # Run the simulation with a focus on the top level (where self-reference happens)
    simulator.simulate(steps=20, strategy="focus_top")
    
    # Visualize the strange loop
    print(simulator.visualize_strange_loop())
    
    # The final trace - the strange loop has modeled itself modeling itself
    # ↻ Recursive traces are forever reflected, never completed ↻
