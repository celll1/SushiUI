"""
TIPO (Text to Image with text Presampling for Optimal prompting) Manager

Provides prompt optimization and expansion using TIPO models.
Based on KBlueLeaf's TIPO framework: https://github.com/KohakuBlueleaf/KGen
"""

import torch
from typing import Optional, Dict, Any
from pathlib import Path


class TIPOManager:
    """Manages TIPO model loading and prompt generation"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False

    def load_model(self, model_name: str = "KBlueLeaf/TIPO-500M"):
        """Load TIPO model

        Args:
            model_name: HuggingFace model ID (default: KBlueLeaf/TIPO-500M)
        """
        try:
            print(f"[TIPO] Loading model: {model_name}")

            # Check if tipo-kgen is installed
            try:
                import kgen.models as models
                import kgen.executor.tipo as tipo_executor
                from kgen.formatter import seperate_tags

                # Store for later use
                self.kgen_models = models
                self.tipo_executor = tipo_executor
                self.seperate_tags = seperate_tags

            except ImportError:
                print("[TIPO] Warning: tipo-kgen package not installed")
                print("[TIPO] Install with: pip install tipo-kgen")
                print("[TIPO] Falling back to transformers-only mode")

                # Fallback to standard transformers
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device
                )
                self.model.eval()

            self.model_name = model_name
            self.loaded = True
            print(f"[TIPO] Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"[TIPO] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.loaded = False

    def generate_prompt(
        self,
        input_prompt: str,
        tag_length: str = "short",
        nl_length: str = "short",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """Generate enhanced prompt from input

        Args:
            input_prompt: Input prompt (tags or natural language)
            tag_length: Length target for tags (very_short/short/long/very_long)
            nl_length: Length target for natural language (very_short/short/long/very_long)
            temperature: Sampling temperature (default: 1.0)
            top_p: Nucleus sampling parameter (default: 0.95)
            top_k: Top-k sampling parameter (default: 50)
            max_new_tokens: Maximum tokens to generate (default: 256)

        Returns:
            Generated/enhanced prompt
        """
        if not self.loaded:
            print("[TIPO] Model not loaded, returning original prompt")
            return input_prompt

        try:
            # Check if using tipo-kgen
            if hasattr(self, 'tipo_executor'):
                return self._generate_with_kgen(
                    input_prompt, tag_length, nl_length,
                    temperature, top_p, top_k, max_new_tokens, **kwargs
                )
            else:
                return self._generate_with_transformers(
                    input_prompt, temperature, top_p, top_k, max_new_tokens, **kwargs
                )

        except Exception as e:
            print(f"[TIPO] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return input_prompt

    def _generate_with_kgen(
        self,
        input_prompt: str,
        tag_length: str,
        nl_length: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        **kwargs
    ) -> str:
        """Generate using tipo-kgen library"""
        try:
            # Parse input as tags or natural language
            # If comma-separated, treat as tags; otherwise as natural language
            if ',' in input_prompt:
                tags = [t.strip() for t in input_prompt.split(',')]
                nl_prompt = ""
            else:
                tags = []
                nl_prompt = input_prompt

            # Parse request
            meta, operations, general, nl_prompt_parsed = self.tipo_executor.parse_tipo_request(
                self.seperate_tags(tags) if tags else [],
                nl_prompt,
                tag_length_target=tag_length,
                generate_extra_nl_prompt=not nl_prompt
            )

            # Run TIPO
            result, timing = self.tipo_executor.tipo_runner(
                meta, operations, general, nl_prompt_parsed,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

            print(f"[TIPO] Generation took {timing:.2f}s")
            return result

        except Exception as e:
            print(f"[TIPO] KGen generation failed: {e}")
            import traceback
            traceback.print_exc()
            return input_prompt

    def _generate_with_transformers(
        self,
        input_prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        **kwargs
    ) -> str:
        """Generate using transformers library directly"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Combine with input if needed
            result = input_prompt + ", " + generated_text if generated_text else input_prompt

            return result

        except Exception as e:
            print(f"[TIPO] Transformers generation failed: {e}")
            import traceback
            traceback.print_exc()
            return input_prompt

    def unload_model(self):
        """Unload TIPO model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.loaded = False
        print("[TIPO] Model unloaded")


# Global TIPO manager instance
tipo_manager = TIPOManager()
