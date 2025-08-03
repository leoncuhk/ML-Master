
import math
from typing import Optional, List, Tuple

class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search.
    Each node corresponds to a specific state of the solution.
    """
    def __init__(self, parent: Optional['MCTSNode'] = None, action: Optional[str] = None):
        self.parent = parent
        self.action = action  # The action that led to this node (e.g., "Draft", "Debug", "Improve")
        self.children: List['MCTSNode'] = []

        # Core MCTS attributes
        self.visit_count: int = 0
        self.total_reward: float = 0.0

        # Solution-specific state
        self.code: Optional[str] = None
        self.execution_feedback: Optional[str] = None
        self.performance_metric: Optional[float] = None
        self.reasoning_insights: Optional[str] = None
        self.is_terminal: bool = False
        self.has_bug: bool = False

        # For termination criteria
        self.failed_improvements: int = 0
        self.debug_depth: int = 0

    def uct_value(self, exploration_constant: float = 1.0) -> float:
        """
        Calculates the UCT (Upper Confidence Bound for Trees) value for this node.
        """
        if self.visit_count == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visit_count == 0:
            # Parent visit count is needed for exploration term, handle root or unvisited parent case
            return self.total_reward / self.visit_count if self.visit_count > 0 else float('inf')

        exploitation_term = self.total_reward / self.visit_count
        exploration_term = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        
        return exploitation_term + exploration_term

    def select_best_child(self, exploration_constant: float) -> 'MCTSNode':
        """
        Selects the child with the highest UCT value.
        """
        return max(self.children, key=lambda child: child.uct_value(exploration_constant))

    def expand(self, action: str):
        """
        Expands the tree with a new child node.
        The actual code and insights will be populated by the reasoning module.
        """
        child_node = MCTSNode(parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, reward: float):
        """
        Propagates the reward up the tree from this node to the root.
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def __repr__(self):
        return f"Node(Action: {self.action}, Q/N: {self.total_reward}/{self.visit_count}, Children: {len(self.children)})"
