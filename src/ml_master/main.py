
import os
import argparse
from .core import MCTSNode
from .explorer import Explorer
from .reasoner import Reasoner

def main():
    """
    Main entry point for the ML-Master agent.
    """
    parser = argparse.ArgumentParser(description="ML-Master: An AI agent for solving machine learning problems.")
    parser.add_argument("--project_name", type=str, default="titanic", help="The name of the project/competition (must be a subdirectory).")
    parser.add_argument("--task_description_file", type=str, default="titanic/task_description.txt", help="Path to the file containing the detailed task description.")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of MCTS iterations to run.")
    args = parser.parse_args()

    # --- Configuration ---
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    try:
        with open(args.task_description_file, 'r') as f:
            task_description = f.read()
    except FileNotFoundError:
        raise ValueError(f"Task description file not found at: {args.task_description_file}")

    # --- Initialization ---
    root_node = MCTSNode()
    reasoner = Reasoner(api_key=api_key)
    # Pass project_name to the explorer to handle project-specific paths
    explorer = Explorer(root=root_node, reasoner=reasoner, task_description=task_description, project_name=args.project_name)

    # --- Main Loop ---
    print(f"Starting ML-Master for project '{args.project_name}' with {args.num_iterations} iterations...")
    for i in range(args.num_iterations):
        print(f"\n--- Iteration {i+1}/{args.num_iterations} ---")
        explorer.run_mcts_cycle()
        print(f"Tree state: {explorer.root}")

    # --- Final Result ---
    print("\nML-Master has finished.")
    # Here you would typically extract the best solution found from the MCTS tree.

if __name__ == "__main__":
    main()

