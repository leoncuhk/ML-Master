
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
        if parent:
            if action == "Debug":
                self.debug_depth = parent.debug_depth + 1
            else:
                # Reset on non-debug actions, including Improve and Draft
                self.debug_depth = 0
            # Inherit failed improvements count from parent, it will be updated later
            self.failed_improvements = parent.failed_improvements


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

    def select_best_child(self, exploration_constant: float) -> Optional['MCTSNode']:
        """
        Selects the child with the highest UCT value that is not terminal.
        """
        non_terminal_children = [child for child in self.children if not child.is_terminal]
        if not non_terminal_children:
            return None
        return max(non_terminal_children, key=lambda child: child.uct_value(exploration_constant))

    def expand(self, action: str) -> 'MCTSNode':
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

    def update_and_check_terminal_status(self, improvement_threshold: float, tau_improve: int, tau_debug: int):
        """
        Updates node status based on the last action and checks if it should become terminal.
        This should be called after a node is evaluated.
        """
        if self.action == "Improve":
            parent_metric = self.parent.performance_metric if self.parent and self.parent.performance_metric is not None else -1.0
            current_metric = self.performance_metric if self.performance_metric is not None else -1.0

            if current_metric - parent_metric < improvement_threshold:
                # Note: failed_improvements is inherited, so we just check against the threshold
                # The count is implicitly managed by the chain of nodes.
                pass # No change to failed_improvements, it's a chain
            else:
                self.failed_improvements = 0 # Reset on successful improvement

        # In __init__, we already increment debug_depth.
        # A new 'Improve' action on a node resets the failed_improvements count for its children.
        if self.parent and self.action == 'Improve':
            if self.performance_metric and self.parent.performance_metric and self.performance_metric > self.parent.performance_metric:
                 self.failed_improvements = 0
            else:
                 self.failed_improvements = self.parent.failed_improvements + 1


        if self.failed_improvements >= tau_improve or self.debug_depth >= tau_debug:
            self.is_terminal = True

    def calculate_reward(self, best_score_so_far: float) -> float:
        """
        Calculates the reward for this node based on the paper's Equation 3.
        This should be called during the 'verification' phase.
        """
        # If the node has defects (e.g., code fails to run), reward is -1.
        if self.has_bug:
            return -1.0

        reward = 0.0

        # Quality reward (r_q): Did the solution improve upon the best metric so far?
        if self.performance_metric is not None and self.performance_metric > best_score_so_far:
            reward += 1.0

        # Debugging reward (r_d): Did we successfully fix a bug from the parent?
        if self.parent and self.parent.has_bug and not self.has_bug:
            reward += 1.0

        # Structural improvement reward (r_s): Was this a successful 'Improve' action?
        # We interpret this as an 'Improve' action that actually increased the score.
        if self.action == "Improve" and self.parent and self.performance_metric is not None and self.parent.performance_metric is not None:
            if self.performance_metric > self.parent.performance_metric:
                reward += 1.0

        return reward


    def __repr__(self):
        return f"Node(Action: {self.action}, Q/N: {self.total_reward:.2f}/{self.visit_count}, Term: {self.is_terminal}, Score: {self.performance_metric:.4f if self.performance_metric else 'N/A'})"
