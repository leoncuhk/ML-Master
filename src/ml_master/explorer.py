from .core import MCTSNode
from .reasoner import Reasoner
from typing import Optional
import subprocess
import os

class Explorer:
    """
    Manages the balanced multi-trajectory exploration process using MCTS.
    This class is designed to be thread-safe for parallel execution.
    """
    def __init__(self, root: MCTSNode, reasoner: Reasoner, task_description: str, project_name: str,
                 exploration_constant: float = 1.0, improvement_threshold: float = 0.001,
                 max_failed_improvements: int = 3, max_debug_depth: int = 20):
        self.root = root
        self.reasoner = reasoner
        self.task_description = task_description
        self.project_name = project_name
        self.exploration_constant = exploration_constant
        self.improvement_threshold = improvement_threshold
        self.max_failed_improvements = max_failed_improvements
        self.max_debug_depth = max_debug_depth
        self.best_performance_metric = -float('inf')

        # Create the submissions directory for the project if it doesn't exist
        self.submissions_dir = os.path.join(project_name, 'generated_submissions')
        os.makedirs(self.submissions_dir, exist_ok=True)

    def _determine_action(self, node: MCTSNode) -> str:
        """Determines the next action based on the node's state, as per the paper's expansion rules."""
        if not node.code:
            return "Draft"
        elif node.has_bug:
            return "Debug"
        else:
            return "Improve"

    def _run_single_mcts_cycle(self):
        """
        Runs a single, complete cycle of MCTS: select, expand, simulate (verify), backpropagate.
        This method is intended to be called in parallel by multiple workers.
        """
        # 1. Selection
        node = self.root
        while True:
            if node.is_terminal:
                # This path is a dead end, so we can't explore further.
                # In a more advanced implementation, we might restart from the root.
                return
            
            best_child = node.select_best_child(self.exploration_constant)
            if best_child is None:
                # Node is not terminal but has no valid children to select, so we expand it.
                break
            node = best_child

        # 2. Expansion
        action_to_perform = self._determine_action(node)
        expanded_node = node.expand(action_to_perform)

        # 3. Reasoning (Simulate)
        solution = self.reasoner.generate_solution_for_action(expanded_node, action_to_perform, self.task_description)
        expanded_node.reasoning_insights = solution["Reasoning"]
        expanded_node.code = solution["Code"]

        # 4. Verification
        self._verify_and_evaluate(expanded_node)

        # 5. Backpropagation
        reward = expanded_node.calculate_reward(self.best_performance_metric)
        expanded_node.backpropagate(reward)
        
        # Update best score if applicable
        if expanded_node.performance_metric is not None and expanded_node.performance_metric > self.best_performance_metric:
            self.best_performance_metric = expanded_node.performance_metric

        # 6. Update termination status
        expanded_node.update_and_check_terminal_status(
            self.improvement_threshold, self.max_failed_improvements, self.max_debug_depth
        )

    def _verify_and_evaluate(self, node: MCTSNode):
        """
        Executes the generated script and evaluates its performance.
        This is the 'verification' phase of the MCTS cycle.
        """
        # Use a unique name for each script to avoid conflicts
        script_filename = os.path.join(self.submissions_dir, f"solution_node_{node.parent.visit_count + 1}_{node.action}.py")
        with open(script_filename, "w") as f:
            f.write(node.code)

        try:
            # This is where a secure sandbox would be critical.
            process = subprocess.run(
                ["uv", "run", "python", script_filename],
                check=True, capture_output=True, text=True, timeout=300
            )
            node.execution_feedback = "Execution successful."
            node.has_bug = False
            
            output = process.stdout
            accuracy_line = [line for line in output.split('\n') if line.startswith("Accuracy:")]
            if not accuracy_line:
                node.execution_feedback += "\nError: Script did not print accuracy in the correct format."
                node.has_bug = True
                return

            score_str = accuracy_line[0].split(':')[1].strip()
            node.performance_metric = float(score_str)

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            node.execution_feedback = f"Execution failed: {e.stderr}"
            node.has_bug = True
        except (IndexError, ValueError) as e:
            node.execution_feedback = f"Scoring failed: Could not parse accuracy. Error: {e}"
            node.has_bug = True
