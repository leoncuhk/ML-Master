from typing import List, Dict, Any, Optional
from .core import MCTSNode
import google.generativeai as genai
import re

class Reasoner:
    """
    Implements the Steerable Reasoning module.
    Interacts with a Gemini LLM and uses adaptive memory to guide the reasoning process.
    """
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro-latest"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def construct_adaptive_memory(self, node: MCTSNode) -> str:
        """
        Constructs the adaptive memory from the parent and sibling nodes.
        """
        memory_parts = []

        # 1. Add context from the parent node
        if node.parent:
            parent = node.parent
            memory_parts.append("### Previous Attempt (Parent Node):\n")
            if parent.reasoning_insights:
                memory_parts.append(f"**Reasoning:** {parent.reasoning_insights}\n")
            if parent.code:
                memory_parts.append(f"**Code:**\n```python\n{parent.code}\n```\n")
            if parent.execution_feedback:
                memory_parts.append(f"**Execution Feedback:** {parent.execution_feedback}\n")
            if parent.performance_metric is not None:
                memory_parts.append(f"**Performance:** {parent.performance_metric}\n")

        # 2. Add context from sibling nodes for contrastive signals
        if node.parent:
            siblings = [child for child in node.parent.children if child is not node]
            if siblings:
                memory_parts.append("\n### Parallel Attempts (Sibling Nodes):\n")
                for i, sibling in enumerate(siblings):
                    memory_parts.append(f"--- Sibling {i+1} ---\n")
                    if sibling.reasoning_insights:
                        memory_parts.append(f"**Reasoning:** {sibling.reasoning_insights}\n")
                    if sibling.performance_metric is not None:
                        memory_parts.append(f"**Performance:** {sibling.performance_metric}\n")

        return "".join(memory_parts)

    def generate_action(self, node: MCTSNode, task_description: str) -> Dict[str, Any]:
        """
        Generates an action (Draft, Debug, Improve) and the corresponding code/plan.
        """
        adaptive_memory = self.construct_adaptive_memory(node)
        
        prompt = f"""
        You are an expert AI developer solving the Kaggle Titanic competition.
        Your task is to write a complete, single Python script to perform the entire machine learning workflow.

        **CONTEXT FROM PREVIOUS ATTEMPTS:**
        {adaptive_memory}

        **INSTRUCTIONS:**
        1.  **Load Data**: Load the training data from 'titanic/train.csv' using pandas.
        2.  **Preprocess & Feature Engineer**: Handle missing values (e.g., for 'Age'), convert categorical features (like 'Sex' and 'Embarked') into numerical format. You can also create new features if you think they are useful.
        3.  **Split Data**: Split the 'train.csv' data into a training set and a validation set (e.g., 80% train, 20% validation). Use `sklearn.model_selection.train_test_split`.
        4.  **Train Model**: Choose a classification model (e.g., LogisticRegression, RandomForestClassifier), and train it on your training set.
        5.  **Evaluate**: Predict on the validation set and calculate the accuracy score.
        6.  **Output**: Your script **MUST** print the final validation accuracy to standard output in the following specific format: `Accuracy: [your_accuracy_score]` where `[your_accuracy_score]` is a float (e.g., `Accuracy: 0.785`). This is the only way your performance can be measured.

        **YOUR ACTION:**
        Based on the context, decide on your action:
        - **Draft**: Write the first version of the script.
        - **Debug**: Fix errors in the previous script.
        - **Improve**: Enhance the performance of the working script.

        **RESPONSE FORMAT:**
        **Action**: [Your chosen action: Draft, Debug, or Improve]
        **Reasoning**: [Your step-by-step thinking process and plan for the script]
        **Code**: [The full Python script that accomplishes all steps above]
        """

        response = self.model.generate_content(prompt)
        parsed_response = self._parse_llm_response(response.text)
        return parsed_response

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parses the LLM's response to extract the action, reasoning, and code.
        """
        try:
            action = re.search(r"\*\*Action\*\*:\s*(Draft|Debug|Improve)", response_text).group(1)
            reasoning = re.search(r"\*\*Reasoning\*\*:\s*([\s\S]*?)\*\*Code\*\*:", response_text).group(1).strip()
            code = re.search(r"\*\*Code\*\*:\s*```python\n([\s\S]*?)```", response_text).group(1).strip()
            return {"Action": action, "Reasoning": reasoning, "Code": code}
        except AttributeError:
            # Fallback if parsing fails
            return {
                "Action": "Debug",
                "Reasoning": "Failed to parse the LLM response. The format might be incorrect.",
                "Code": "# Parsing failed, please check the LLM output format.\nprint('Error: Could not parse LLM response.')"
            }
