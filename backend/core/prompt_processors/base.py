"""Base class for prompt processors

This provides an extensible framework for processing special prompt syntax.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class PromptProcessor(ABC):
    """Base class for prompt processors

    Prompt processors handle special syntax in prompts, such as:
    - Prompt editing: [from:to:when]
    - Alternating words: [word1|word2|word3]
    - Composable diffusion: AND syntax
    - etc.

    Each processor can:
    1. Parse the prompt and extract special syntax
    2. Provide step callbacks to modify prompts during generation
    3. Return the processed/cleaned prompt
    """

    def __init__(self):
        self.enabled = True

    @abstractmethod
    def parse(self, prompt: str, num_steps: int) -> Dict[str, Any]:
        """Parse the prompt and extract processor-specific information

        Args:
            prompt: The input prompt with special syntax
            num_steps: Total number of inference steps

        Returns:
            Dictionary containing:
            - 'initial_prompt': The prompt to use at step 0
            - 'edits': List of edits to apply at specific steps
            - Any other processor-specific data
        """
        pass

    @abstractmethod
    def get_prompt_at_step(self, step: int, total_steps: int) -> Optional[str]:
        """Get the prompt that should be used at a specific step

        Args:
            step: Current step number (0-indexed)
            total_steps: Total number of steps

        Returns:
            The prompt to use at this step, or None if no change
        """
        pass

    def reset(self):
        """Reset the processor state for a new generation"""
        pass
