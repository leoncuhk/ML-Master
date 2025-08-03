from .core import MCTSNode
from .reasoner import Reasoner
from typing import Optional
import subprocess
import os

class Explorer:
    """
    Manages the balanced multi-trajectory exploration process using MCTS.
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

    def select_node(self) -> MCTSNode:
        """
        Selects a node to expand using the MCTS selection policy.
        """
        node = self.root
        while node.children:
            node = node.select_best_child(self.exploration_constant)
            if node.is_terminal:
                return node # Return the terminal node to stop expansion
        return node

    def run_mcts_cycle(self):
        """
        Runs a single cycle of MCTS: select, expand, simulate (verify), backpropagate.
        """
        # 1. Selection
        selected_node = self.select_node()

        if selected_node.is_terminal:
            return # Don't expand terminal nodes

        # 2. Expansion (and Reasoning)
        action_plan = self.reasoner.generate_action(selected_node, self.task_description)
        
        action = action_plan["Action"]
        reasoning = action_plan["Reasoning"]
        code = action_plan["Code"]

        expanded_node = selected_node.expand(action)
        expanded_node.reasoning_insights = reasoning
        expanded_node.code = code

        # Update debug depth
        if action == "Debug":
            expanded_node.debug_depth = selected_node.debug_depth + 1

        # 3. Verification (Simulate)
        reward = self._simulate_and_get_reward(expanded_node)

        # 4. Backpropagation
        expanded_node.backpropagate(reward)

        # 5. Update termination status
        self._update_termination_status(expanded_node)

    def _simulate_and_get_reward(self, node: MCTSNode) -> float:
        """
        Executes the generated script for the Titanic problem and calculates the reward.
        The script is expected to train a model and print the validation accuracy.
        """
        # 1. Write the generated code to a temporary script file
        # Use a unique name for each script to avoid conflicts if we run in parallel later
        script_filename = os.path.join(self.submissions_dir, f"solution_node_{node.parent.visit_count + 1}.py")
        with open(script_filename, "w") as f:
            f.write(node.code)

        # 2. Run the generated script
        try:
            process = subprocess.run(
                ["uv", "run", "python", script_filename],
                check=True, capture_output=True, text=True, timeout=300 # 5 minute timeout
            )
            node.execution_feedback = "Execution successful."
            node.has_bug = False
            
            # 3. Parse the accuracy from the script's output
            output = process.stdout
            accuracy_line = [line for line in output.split('\n') if line.startswith("Accuracy:")]
            if not accuracy_line:
                node.execution_feedback += "\nError: Script did not print accuracy in the correct format."
                node.has_bug = True
                return -1 # Penalize for not following instructions

            score = float(accuracy_line[0].split(':')[1].strip())
            node.performance_metric = score

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            error_message = e.stderr
            # Try to extract a more specific error message
            if "TypeError:" in error_message:
                error_message = [line for line in error_message.split('\n') if "TypeError:" in line][-1]
            elif "ModuleNotFoundError:" in error_message:
                error_message = [line for line in error_message.split('\n') if "ModuleNotFoundError:" in line][-1]
            elif "SyntaxError:" in error_message:
                error_message = [line for line in error_message.split('\n') if "SyntaxError:" in line][-1]

            node.execution_feedback = f"Execution failed: {error_message}"
            node.has_bug = True
            return -1 # Penalize failed executions
        except (IndexError, ValueError) as e:
            node.execution_feedback = f"\nScoring failed: Could not parse accuracy from script output. Error: {e}"
            node.has_bug = True
            return -1

        # 4. Calculate reward based on the score (Equation 3 from the paper)
        reward = 0
        # r_q: Quality reward - higher accuracy is better
        if node.performance_metric > (self.best_performance_metric or -1):
            reward += 1
            self.best_performance_metric = node.performance_metric
        
        # r_d: Debugging reward
        if node.parent and node.parent.has_bug and not node.has_bug:
            reward += 1

        # r_s: Structural improvement reward
        if node.action == "Improve" and node.performance_metric > (node.parent.performance_metric or -1):
            reward += 1

        return reward

    def _update_termination_status(self, node: MCTSNode):
        """
        Checks and updates the termination status of a node based on the criteria in the paper.
        """
        # Debug depth constraint
        if node.debug_depth >= self.max_debug_depth:
            node.is_terminal = True
            return

        # Improvement-based termination
        if node.action == "Improve":
            # If the performance metric could not be calculated, it's a failed improvement.
            if node.performance_metric is None:
                node.failed_improvements = (node.parent.failed_improvements or 0) + 1
            else:
                parent_metric = node.parent.performance_metric or -float('inf')
                if node.performance_metric - parent_metric < self.improvement_threshold:
                    node.failed_improvements = (node.parent.failed_improvements or 0) + 1
                else:
                    node.failed_improvements = 0
            
            if node.failed_improvements >= self.max_failed_improvements:
                node.is_terminal = True