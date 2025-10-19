"""Prompt editing processor

Implements A1111-style prompt editing syntax:
- [from:to:when] - swap from 'from' to 'to' at step 'when'
- [to:when] - add 'to' at step 'when'
- [from::when] - remove 'from' at step 'when'

Where 'when' can be:
- A value < 1.0 (e.g., 0.5): fraction of total steps (0.5 = 50%)
- A value >= 1.0 (e.g., 15): absolute step number

Examples:
- [cat:dog:0.5] - switch from 'cat' to 'dog' at 50% of total steps (step 15 if 30 steps)
- [cat:dog:15] - switch from 'cat' to 'dog' at step 15
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from .base import PromptProcessor


class PromptEdit:
    """Represents a single prompt edit"""

    def __init__(self, from_text: str, to_text: str, when: float, is_fractional: bool):
        self.from_text = from_text
        self.to_text = to_text
        self.when = when
        self.is_fractional = is_fractional

    def get_step(self, total_steps: int) -> int:
        """Calculate the absolute step number when this edit should be applied"""
        if self.is_fractional:
            return int(self.when * total_steps)
        else:
            return int(self.when)

    def __repr__(self):
        return f"PromptEdit(from='{self.from_text}', to='{self.to_text}', when={self.when}, fractional={self.is_fractional})"


class PromptEditingProcessor(PromptProcessor):
    """Processes prompt editing syntax"""

    def __init__(self):
        super().__init__()
        self.original_prompt = ""
        self.current_prompt = ""
        self.edits: List[Tuple[int, PromptEdit]] = []  # (step, edit)
        self.total_steps = 0

    def parse(self, prompt: str, num_steps: int) -> Dict[str, Any]:
        """Parse prompt editing syntax

        Args:
            prompt: The prompt with editing syntax
            num_steps: Total number of inference steps

        Returns:
            Dictionary with initial_prompt and edits
        """
        self.original_prompt = prompt
        self.total_steps = num_steps
        self.edits = []

        # Parse and extract all prompt editing patterns
        parsed_prompt, edits = self._parse_recursive(prompt, num_steps)

        self.current_prompt = parsed_prompt
        self.edits = sorted(edits, key=lambda x: x[0])  # Sort by step

        print(f"[PromptEditing] Parsed prompt: '{parsed_prompt}'")
        print(f"[PromptEditing] Found {len(self.edits)} edits:")
        for step, edit in self.edits:
            print(f"  Step {step}: '{edit.from_text}' -> '{edit.to_text}'")

        return {
            'initial_prompt': parsed_prompt,
            'edits': self.edits,
            'total_steps': num_steps
        }

    def _parse_recursive(self, prompt: str, num_steps: int, depth: int = 0) -> Tuple[str, List[Tuple[int, PromptEdit]]]:
        """Recursively parse prompt editing syntax (handles nesting)

        Args:
            prompt: Prompt to parse
            num_steps: Total steps
            depth: Current recursion depth (for debugging)

        Returns:
            Tuple of (processed_prompt, list of (step, edit) tuples)
        """
        if depth > 10:  # Prevent infinite recursion
            print(f"[PromptEditing] Warning: Max recursion depth reached")
            return prompt, []

        edits = []
        result = prompt

        # Pattern for prompt editing: [from:to:when] or [to:when] or [from::when]
        # This regex matches the outermost brackets first
        pattern = r'\[([^\[\]]*):([^\[\]]*):([0-9.]+)\]|\[([^\[\]]*):([0-9.]+)\]|\[([^\[\]]*):():([0-9.]+)\]'

        while True:
            match = re.search(pattern, result)
            if not match:
                break

            full_match = match.group(0)
            start_pos = match.start()
            end_pos = match.end()

            # Determine which pattern matched
            if match.group(1) is not None:  # [from:to:when]
                from_text = match.group(1)
                to_text = match.group(2)
                when_str = match.group(3)
            elif match.group(4) is not None:  # [to:when]
                from_text = ""
                to_text = match.group(4)
                when_str = match.group(5)
            else:  # [from::when]
                from_text = match.group(6)
                to_text = ""
                when_str = match.group(8)

            # Parse 'when' value
            when_value = float(when_str)
            # If value is < 1.0, treat as fractional (0.0-1.0 = 0%-100%)
            # If value is >= 1.0, treat as absolute step number
            is_fractional = when_value < 1.0

            # Create edit
            edit = PromptEdit(from_text, to_text, when_value, is_fractional)
            step = edit.get_step(num_steps)
            edits.append((step, edit))

            # Replace the matched pattern with the 'from' text initially
            result = result[:start_pos] + from_text + result[end_pos:]

        return result, edits

    def get_prompt_at_step(self, step: int, total_steps: int) -> Optional[str]:
        """Get the prompt that should be used at a specific step

        Args:
            step: Current step (0-indexed)
            total_steps: Total number of steps

        Returns:
            The modified prompt, or None if no changes at this step
        """
        # Check if any edits should be applied at this step
        prompt_changed = False
        current = self.current_prompt

        for edit_step, edit in self.edits:
            if edit_step == step:
                # Apply this edit
                if edit.from_text:
                    # Replace from_text with to_text
                    current = current.replace(edit.from_text, edit.to_text)
                else:
                    # Just add to_text
                    current = current + " " + edit.to_text

                prompt_changed = True
                print(f"[PromptEditing] Step {step}: Applied edit '{edit.from_text}' -> '{edit.to_text}'")
                print(f"[PromptEditing] New prompt: '{current}'")

        if prompt_changed:
            self.current_prompt = current
            return current

        return None

    def reset(self):
        """Reset for a new generation"""
        self.current_prompt = ""
        self.edits = []
        self.total_steps = 0
