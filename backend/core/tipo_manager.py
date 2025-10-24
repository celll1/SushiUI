"""
TIPO (Text to Image with text Presampling for Optimal prompting) Manager

Provides prompt optimization and expansion using TIPO models.
Based on KBlueLeaf's TIPO framework: https://github.com/KohakuBlueleaf/KGen
"""

import torch
import re
from typing import Optional, Dict, Any, List
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
            print(f"[TIPO Transform] Input: '{input_prompt}'")
            print(f"[TIPO Transform] max_new_tokens: {max_new_tokens}, temperature: {temperature}")

            # Tokenize input
            inputs = self.tokenizer(
                input_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            print(f"[TIPO Transform] Input token length: {inputs['input_ids'].shape[1]}")

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

            print(f"[TIPO Transform] Output token length: {outputs.shape[1]}")

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            print(f"[TIPO Transform] Generated text length: {len(generated_text)} chars")
            print(f"[TIPO Transform] Generated text preview: {generated_text[:200]}...")

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

    def parse_tipo_output(self, output: str) -> Dict[str, Any]:
        """Parse TIPO output into structured format

        Args:
            output: Raw TIPO output string

        Returns:
            Dictionary with parsed components:
            {
                'rating': str,
                'artist': str,
                'copyright': str,
                'characters': str,
                'target': str,
                'short_nl': str,
                'long_nl': str,
                'tags': List[str],
                'special_tags': List[str],  # 1girl, 1boy, etc.
                'quality_tags': List[str],
                'meta_tags': List[str],
                'general_tags': List[str]
            }
        """
        result = {
            'rating': '',
            'artist': '',
            'copyright': '',
            'characters': '',
            'target': '',
            'short_nl': '',
            'long_nl': '',
            'tags': [],
            'special_tags': [],
            'quality_tags': [],
            'meta_tags': [],
            'general_tags': []
        }


        # Parse line-by-line
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse key-value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()

                if key == 'artrating' or key == 'rating':
                    result['rating'] = value
                elif key == 'artist':
                    result['artist'] = value
                elif key == 'copyright' or key == 'copyrights':
                    result['copyright'] = value
                elif key == 'characters' or key == 'character':
                    result['characters'] = value
                elif key == 'target':
                    result['target'] = value
                elif key == 'short':
                    result['short_nl'] = value
                elif key == 'long':
                    result['long_nl'] = value
                elif key == 'tag' or key == 'tags':
                    # Parse comma-separated tags
                    tags = [t.strip() for t in value.split(',') if t.strip()]
                    result['tags'] = tags

                    # Categorize tags according to TIPO's classification
                    result['special_tags'] = self._extract_special_tags(tags)
                    result['quality_tags'] = self._extract_quality_tags(tags)
                    result['rating_tags'] = self._extract_rating_tags(tags)
                    result['meta_tags'] = self._extract_meta_tags(tags)

                    # General tags = everything not in other categories
                    used_tags = (result['special_tags'] + result['quality_tags'] +
                                result['rating_tags'] + result['meta_tags'])
                    result['general_tags'] = self._extract_general_tags(tags, used_tags)

                    # Debug logging
                    print(f"[TIPO Parse] Total tags: {len(tags)}")
                    print(f"[TIPO Parse] Special: {result['special_tags']}")
                    print(f"[TIPO Parse] Quality: {result['quality_tags']}")
                    print(f"[TIPO Parse] Rating: {result['rating_tags']}")
                    print(f"[TIPO Parse] Meta: {result['meta_tags']}")
                    print(f"[TIPO Parse] General: {len(result['general_tags'])} tags")

        return result

    def _extract_special_tags(self, tags: List[str]) -> List[str]:
        """Extract special tags (character count: 1girl, 1boy, 2girls, etc.)

        TIPO's <|special|> category includes character count tags.
        """
        special_patterns = [
            r'^\d+girl(s)?$',     # 1girl, 2girls, etc.
            r'^\d+boy(s)?$',      # 1boy, 2boys, etc.
            r'^\d+other(s)?$',    # 1other, 2others, etc.
            r'^solo$',            # solo
            r'^multiple girls$',  # multiple girls
            r'^multiple boys$',   # multiple boys
        ]

        special_tags = []
        for tag in tags:
            tag_lower = tag.lower()
            for pattern in special_patterns:
                if re.match(pattern, tag_lower):
                    special_tags.append(tag)
                    break
        return special_tags

    def _extract_quality_tags(self, tags: List[str]) -> List[str]:
        """Extract quality-related tags

        TIPO's <|quality|> category includes score_xxx and quality descriptors.
        Note: This should NOT include meta tags like highres/lowres/absurdres
        """
        quality_patterns = [
            r'^score_\d+',              # score_9, score_8, etc.
            r'^.*\s+quality$',          # best quality, high quality, etc. (with space)
            r'^quality$',               # quality (single word)
            r'^masterpiece$',
            r'^amazing\s+quality$',
            r'^great\s+quality$',
            r'^worst\s+quality$',
        ]

        quality_tags = []
        for tag in tags:
            tag_lower = tag.lower()
            for pattern in quality_patterns:
                if re.match(pattern, tag_lower):
                    quality_tags.append(tag)
                    print(f"[TIPO] Matched quality tag: '{tag}' with pattern '{pattern}'")
                    break
        return quality_tags

    def _extract_meta_tags(self, tags: List[str]) -> List[str]:
        """Extract meta tags

        TIPO's <|meta|> category includes highres/lowres/absurdres and year tags.
        """
        meta_keywords = ['highres', 'lowres', 'absurdres']
        meta_patterns = [
            r'^\d{4}$',  # Year tags (4 digits)
        ]

        meta_tags = []
        for tag in tags:
            tag_lower = tag.lower()

            # Check keywords
            if tag_lower in meta_keywords:
                meta_tags.append(tag)
                continue

            # Check patterns
            for pattern in meta_patterns:
                if re.match(pattern, tag):
                    meta_tags.append(tag)
                    break

        return meta_tags

    def _extract_rating_tags(self, tags: List[str]) -> List[str]:
        """Extract rating tags

        TIPO's <|rating|> category includes rating tags like general, sensitive, nsfw, explicit
        """
        rating_keywords = [
            'general', 'sensitive', 'questionable', 'explicit', 'nsfw', 'safe',
            'rating:general', 'rating:sensitive', 'rating:questionable', 'rating:explicit'
        ]

        rating_tags = []
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in rating_keywords:
                rating_tags.append(tag)

        return rating_tags

    def _extract_general_tags(self, tags: List[str], exclude: List[str]) -> List[str]:
        """Extract general tags (everything not in other categories)"""
        return [t for t in tags if t not in exclude]

    def parse_input_tags(self, input_prompt: str) -> Dict[str, Any]:
        """Parse input prompt tags into categories

        Args:
            input_prompt: Original input prompt

        Returns:
            Dictionary with categorized input tags
        """
        # Split input into tags (comma-separated)
        input_tags = [t.strip() for t in input_prompt.split(',') if t.strip()]

        # Categorize input tags
        result = {
            'special_tags': self._extract_special_tags(input_tags),
            'quality_tags': self._extract_quality_tags(input_tags),
            'rating_tags': self._extract_rating_tags(input_tags),
            'meta_tags': self._extract_meta_tags(input_tags),
        }

        used_tags = (result['special_tags'] + result['quality_tags'] +
                    result['rating_tags'] + result['meta_tags'])
        result['general_tags'] = self._extract_general_tags(input_tags, used_tags)

        print(f"[TIPO] Input tags parsed: special={result['special_tags']}, quality={result['quality_tags']}, rating={result['rating_tags']}, meta={result['meta_tags']}, general={len(result['general_tags'])}")

        return result

    def merge_tags(self, input_parsed: Dict[str, Any], tipo_parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Merge input tags with TIPO generated tags

        Args:
            input_parsed: Parsed input tags
            tipo_parsed: Parsed TIPO output

        Returns:
            Merged tags dictionary
        """
        merged = {
            'rating': tipo_parsed.get('rating', ''),
            'artist': tipo_parsed.get('artist', ''),
            'copyright': tipo_parsed.get('copyright', ''),
            'characters': tipo_parsed.get('characters', ''),
            'target': tipo_parsed.get('target', ''),
            'short_nl': tipo_parsed.get('short_nl', ''),
            'long_nl': tipo_parsed.get('long_nl', ''),
        }

        # Global deduplication across ALL categories and fields (case-insensitive)
        # This ensures that a tag like "original" doesn't appear in both general_tags and copyright field
        seen_lower = set()

        # First, collect all tags from string fields (artist, copyright, characters) to avoid duplicates
        for field in ['artist', 'copyright', 'characters']:
            field_value = tipo_parsed.get(field, '')
            if field_value:
                # Split by comma if multiple values
                field_tags = [t.strip() for t in field_value.split(',') if t.strip()]
                for tag in field_tags:
                    seen_lower.add(tag.lower())

        # Merge tags from both sources, preserving order and removing duplicates
        for category in ['special_tags', 'quality_tags', 'rating_tags', 'meta_tags', 'general_tags']:
            input_tags = input_parsed.get(category, [])
            tipo_tags = tipo_parsed.get(category, [])

            combined = []

            # Add input tags first (preserve user's original tags)
            for tag in input_tags:
                tag_lower = tag.lower()
                if tag_lower not in seen_lower:
                    seen_lower.add(tag_lower)
                    combined.append(tag)

            # Add TIPO tags (skip if already seen in ANY category)
            for tag in tipo_tags:
                tag_lower = tag.lower()
                if tag_lower not in seen_lower:
                    seen_lower.add(tag_lower)
                    combined.append(tag)

            merged[category] = combined

        print(f"[TIPO] Merged tags: special={len(merged['special_tags'])}, quality={len(merged['quality_tags'])}, rating={len(merged['rating_tags'])}, meta={len(merged['meta_tags'])}, general={len(merged['general_tags'])}")

        return merged

    def format_prompt_from_parsed(
        self,
        parsed: Dict[str, Any],
        order: List[str],
        enabled_categories: Dict[str, bool]
    ) -> str:
        """Format prompt from parsed TIPO output according to user preferences

        Args:
            parsed: Parsed TIPO output from parse_tipo_output()
            order: List of category names in desired order
                   (e.g., ['special', 'quality', 'rating', 'artist', 'copyright',
                    'characters', 'meta', 'general', 'short_nl', 'long_nl'])
            enabled_categories: Dict mapping category names to whether they're enabled

        Returns:
            Formatted prompt string
        """
        parts = []

        for category in order:
            if not enabled_categories.get(category, True):
                continue

            if category == 'special' and parsed.get('special_tags'):
                parts.extend(parsed['special_tags'])
            elif category == 'quality' and parsed.get('quality_tags'):
                parts.extend(parsed['quality_tags'])
            elif category == 'rating':
                # Check both rating (string from TIPO) and rating_tags (list from input/tags)
                # Use rating_tags (from tag categorization) in priority, fall back to rating string
                if parsed.get('rating_tags'):
                    parts.extend(parsed['rating_tags'])
                elif parsed.get('rating'):
                    parts.append(parsed['rating'])
            elif category == 'artist' and parsed.get('artist'):
                parts.append(parsed['artist'])
            elif category == 'copyright' and parsed.get('copyright'):
                parts.append(parsed['copyright'])
            elif category == 'characters' and parsed.get('characters'):
                parts.append(parsed['characters'])
            elif category == 'meta' and parsed.get('meta_tags'):
                parts.extend(parsed['meta_tags'])
            elif category == 'general' and parsed.get('general_tags'):
                parts.extend(parsed['general_tags'])
            elif category == 'short_nl' and parsed.get('short_nl'):
                parts.append(parsed['short_nl'])
            elif category == 'long_nl' and parsed.get('long_nl'):
                parts.append(parsed['long_nl'])

        return ', '.join(parts)


# Global TIPO manager instance
tipo_manager = TIPOManager()
