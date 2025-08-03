
import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from .core import MCTSNode
from .explorer import Explorer
from .reasoner import Reasoner

def main():
    """
    Main entry point for the ML-Master agent.
    Initializes the system and runs the parallel MCTS exploration.
    """
    load_dotenv()  # Load environment variables from .env file
    parser = argparse.ArgumentParser(description="ML-Master: An AI agent for solving machine learning problems.")
    parser.add_argument("--project_name", type=str, default="titanic", help="The name of the project/competition (must be a subdirectory).")
    parser.add_argument("--task_description_file", type=str, default="titanic/task_description.txt", help="Path to the file containing the detailed task description.")
    parser.add_argument("--num_iterations", type=int, default=20, help="Total number of MCTS cycles to run in parallel.")
    parser.add_argument("--parallelism", type=int, default=3, help="Number of parallel MCTS cycles to run.")
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
    explorer = Explorer(
        root=root_node, 
        reasoner=reasoner, 
        task_description=task_description, 
        project_name=args.project_name
    )

    # --- Parallel MCTS Main Loop ---
    print(f"Starting ML-Master for project '{args.project_name}' with {args.num_iterations} iterations across {args.parallelism} workers...")
    
    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = [executor.submit(explorer._run_single_mcts_cycle) for _ in range(args.num_iterations)]
        
        for i, future in enumerate(as_completed(futures)):
            try:
                future.result() # We don't expect a return value, but this will raise exceptions if they occurred
                print(f"Completed MCTS cycle {i+1}/{args.num_iterations}. Best score so far: {explorer.best_performance_metric:.4f}")
            except Exception as e:
                print(f"An error occurred in an MCTS cycle: {e}")

    # --- Final Result ---
    print("\nML-Master has finished.")
    print(f"Final best score achieved: {explorer.best_performance_metric:.4f}")
    # In a real scenario, you would find the best node and present its code as the final solution.

if __name__ == "__main__":
    main()

