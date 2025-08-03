from typing import List, Dict, Any, Optional
from .core import MCTSNode
import google.generativeai as genai
import re
import time
import random
from google.api_core import exceptions

class Reasoner:
    """
    Implements the Steerable Reasoning module.
    Interacts with a Gemini LLM and uses adaptive memory to guide the reasoning process.
    This module is given a specific action (Draft, Debug, Improve) and generates a solution.
    """
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro-latest"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def _construct_adaptive_memory(self, node: MCTSNode) -> str:
        """
        Constructs the adaptive memory from the parent and sibling nodes,
        implementing the logic from Equation 7 of the ML-Master paper.
        """
        memory_parts = []

        # 1. Context from the immediate parent node (t-1)
        if node.parent:
            parent = node.parent
            memory_parts.append("### Context from Previous Attempt (Parent Node):\n")
            if parent.reasoning_insights:
                memory_parts.append(f"**Reasoning:** {parent.reasoning_insights}\n")
            if parent.code:
                memory_parts.append(f"**Code:**\n```python\n{parent.code}\n```\n")
            if parent.execution_feedback:
                memory_parts.append(f"**Execution Feedback:** {parent.execution_feedback}\n")
            if parent.performance_metric is not None:
                memory_parts.append(f"**Performance:** {parent.performance_metric}\n")

        # 2. Context from sibling nodes (s in S_t) for contrastive signals
        if node.parent:
            siblings = [child for child in node.parent.children if child is not node]
            if siblings:
                memory_parts.append("\n### Context from Parallel Attempts (Sibling Nodes):\n")
                for i, sibling in enumerate(siblings):
                    # Only include siblings that have been evaluated
                    if sibling.visit_count > 0:
                        memory_parts.append(f"--- Sibling Attempt {i+1} ---\n")
                        if sibling.reasoning_insights:
                            memory_parts.append(f"**Reasoning:** {sibling.reasoning_insights}\n")
                        if sibling.performance_metric is not None:
                            memory_parts.append(f"**Performance:** {sibling.performance_metric}\n")
                        if sibling.has_bug:
                            memory_parts.append(f"**Result:** This attempt resulted in a bug.\n")

        if not memory_parts:
            return "No previous attempts. This is the first draft."

        return "".join(memory_parts)

    def generate_solution_for_action(self, node: MCTSNode, action: str, task_description: str) -> Dict[str, Any]:
        """
        Generates a solution for a given action (Draft, Debug, Improve).
        The reasoning process is steered by the adaptive memory.
        Includes retry logic with exponential backoff for API calls.
        """
        adaptive_memory = self._construct_adaptive_memory(node)

        prompt_template = f"""
        You are an expert AI developer. Your goal is to solve a machine learning task by writing a complete, single Python script.

        **TASK DESCRIPTION:**
        {task_description}

        **CONTEXT FROM PREVIOUS EXPLORATION (ADAPTIVE MEMORY):**
        {adaptive_memory}

        **YOUR ASSIGNED ACTION: {action}**

        **INSTRUCTIONS:**
        Based on the task description, the adaptive memory, and your assigned action, please provide your reasoning and the full Python code.

        - If the action is **Draft**: Write the first version of the script from scratch, keeping the task description in mind.
        - If the action is **Debug**: Analyze the provided code and execution feedback from the parent node. Identify the bug and provide a corrected, complete script.
        - If the action is **Improve**: Analyze the parent node's solution. Propose a specific enhancement (e.g., change model, add feature engineering, tune hyperparameters) and provide the complete, improved script.

        **RESPONSE FORMAT:**
        **Reasoning**: [Your step-by-step thinking process for the assigned action. Explain how you are using the adaptive memory to inform your decisions.]
        **Code**: [The full Python script.]
        """

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt_template)
                return self._parse_llm_response(response.text)
            except exceptions.ResourceExhausted as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.random()
                    print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e  # Re-raise the exception after the last attempt
        
        # This part should not be reachable if the loop raises an exception
        return {
            "Reasoning": "Failed to get a response from the LLM after multiple retries.",
            "Code": "# LLM call failed repeatedly due to rate limiting or other issues."
        }


    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parses the LLM's response to extract the reasoning and code.
        """
        try:
            # Reasoning is between "**Reasoning**:" and "**Code**:")
            reasoning_match = re.search(r"\*\*Reasoning\*\*:\s*([\s\S]*?)(?=\n\*\*Code\*\*:)", response_text)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            # Code is inside the python block
            code_match = re.search(r"\*\*Code\*\*:\s*```python\n([\s\S]*?)```", response_text)
            code = code_match.group(1).strip() if code_match else ""

            if not code:
                raise AttributeError("Code block not found in response.")

            return {"Reasoning": reasoning, "Code": code}
        except AttributeError as e:
            # Fallback if parsing fails
            return {
                "Reasoning": f"Failed to parse the LLM response: {e}. The format might be incorrect.",
                "Code": f"# Parsing failed, please check the LLM output format.\nprint('Error: Could not parse LLM response.')\n# Raw response:\n# {response_text}"
            }

