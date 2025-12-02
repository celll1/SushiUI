"""
Optimizer Factory for LoRA Training

Provides a centralized factory for creating optimizers with support for:
- PyTorch native optimizers (AdamW, Adafactor)
- bitsandbytes 8-bit optimizers (AdamW8bit, Lion8bit)
- bitsandbytes paged optimizers (PagedAdamW, PagedAdamW8bit, PagedLion8bit)
"""

import torch
from typing import List, Dict, Any, Optional


class OptimizerFactory:
    """
    Factory class for creating optimizers.

    Supported optimizers:
    - adamw: PyTorch AdamW (32bit)
    - adamw8bit: bitsandbytes AdamW8bit (8bit)
    - paged_adamw: bitsandbytes PagedAdamW (32bit with CPU offloading)
    - paged_adamw8bit: bitsandbytes PagedAdamW8bit (8bit with CPU offloading)
    - adafactor: PyTorch Adafactor (adaptive learning rate, no momentum)
    - lion8bit: bitsandbytes Lion8bit (8bit)
    - paged_lion8bit: bitsandbytes PagedLion8bit (8bit with CPU offloading)
    """

    @staticmethod
    def get_available_optimizers() -> List[str]:
        """
        Get list of available optimizers.

        Returns:
            List of optimizer names
        """
        optimizers = ["adamw", "adamw8bit", "adafactor", "lion8bit"]

        # Check if bitsandbytes is available for paged optimizers
        try:
            import bitsandbytes as bnb
            optimizers.extend(["paged_adamw", "paged_adamw8bit", "paged_lion8bit"])
        except ImportError:
            pass

        return optimizers

    @staticmethod
    def create_optimizer(
        optimizer_type: str,
        params: List,
        learning_rate: float,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        Create optimizer instance.

        Args:
            optimizer_type: Type of optimizer (adamw, adamw8bit, adafactor, lion8bit, etc.)
            params: List of parameters to optimize
            learning_rate: Learning rate
            weight_decay: Weight decay coefficient
            betas: Betas for Adam-based optimizers
            eps: Epsilon for numerical stability
            **kwargs: Additional optimizer-specific arguments

        Returns:
            Optimizer instance

        Raises:
            ValueError: If optimizer_type is unknown
            ImportError: If required library is not available
        """
        optimizer_type = optimizer_type.lower()

        # PyTorch native optimizers
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=learning_rate,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps,
            )
            print(f"[OptimizerFactory] Created AdamW optimizer (32bit)")
            return optimizer

        elif optimizer_type == "adafactor":
            # PyTorch native Adafactor
            # Parameters: lr, beta2_decay, eps, d, weight_decay, foreach, maximize
            optimizer = torch.optim.Adafactor(
                params,
                lr=learning_rate,
                beta2_decay=-0.8,
                eps=(1e-30, 1e-3),
                d=1.0,
                weight_decay=weight_decay,
                foreach=False,
            )
            print(f"[OptimizerFactory] Created Adafactor optimizer")
            return optimizer

        # bitsandbytes optimizers
        elif optimizer_type in ["adamw8bit", "paged_adamw", "paged_adamw8bit", "lion8bit", "paged_lion8bit"]:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    f"bitsandbytes library is required for '{optimizer_type}' optimizer. "
                    "Install with: pip install bitsandbytes"
                )

            if optimizer_type == "adamw8bit":
                optimizer = bnb.optim.AdamW8bit(
                    params,
                    lr=learning_rate,
                    betas=betas,
                    weight_decay=weight_decay,
                    eps=eps,
                )
                print(f"[OptimizerFactory] Created AdamW8bit optimizer")
                return optimizer

            elif optimizer_type == "paged_adamw":
                optimizer = bnb.optim.PagedAdamW(
                    params,
                    lr=learning_rate,
                    betas=betas,
                    weight_decay=weight_decay,
                    eps=eps,
                )
                print(f"[OptimizerFactory] Created PagedAdamW optimizer")
                return optimizer

            elif optimizer_type == "paged_adamw8bit":
                optimizer = bnb.optim.PagedAdamW8bit(
                    params,
                    lr=learning_rate,
                    betas=betas,
                    weight_decay=weight_decay,
                    eps=eps,
                )
                print(f"[OptimizerFactory] Created PagedAdamW8bit optimizer")
                return optimizer

            elif optimizer_type == "lion8bit":
                # Lion optimizer uses different default betas
                lion_betas = kwargs.get("lion_betas", (0.9, 0.99))
                optimizer = bnb.optim.Lion8bit(
                    params,
                    lr=learning_rate,
                    betas=lion_betas,
                    weight_decay=weight_decay,
                )
                print(f"[OptimizerFactory] Created Lion8bit optimizer")
                return optimizer

            elif optimizer_type == "paged_lion8bit":
                # Lion optimizer uses different default betas
                lion_betas = kwargs.get("lion_betas", (0.9, 0.99))
                optimizer = bnb.optim.PagedLion8bit(
                    params,
                    lr=learning_rate,
                    betas=lion_betas,
                    weight_decay=weight_decay,
                )
                print(f"[OptimizerFactory] Created PagedLion8bit optimizer")
                return optimizer

        else:
            raise ValueError(
                f"Unknown optimizer type: '{optimizer_type}'. "
                f"Available optimizers: {', '.join(OptimizerFactory.get_available_optimizers())}"
            )

    @staticmethod
    def get_optimizer_info(optimizer_type: str) -> Dict[str, Any]:
        """
        Get information about an optimizer.

        Args:
            optimizer_type: Type of optimizer

        Returns:
            Dictionary with optimizer information:
            - name: Human-readable name
            - description: Description
            - memory_reduction: Approximate memory reduction (%)
            - requires_bitsandbytes: Whether bitsandbytes is required
            - supports_paging: Whether CPU paging is supported
        """
        optimizer_info = {
            "adamw": {
                "name": "AdamW (32bit)",
                "description": "PyTorch AdamW optimizer (32-bit)",
                "requires_bitsandbytes": False,
                "supports_paging": False,
            },
            "adamw8bit": {
                "name": "AdamW 8bit",
                "description": "bitsandbytes AdamW optimizer (8-bit quantization)",
                "requires_bitsandbytes": True,
                "supports_paging": False,
            },
            "paged_adamw": {
                "name": "Paged AdamW (32bit)",
                "description": "bitsandbytes PagedAdamW (32-bit with CPU offloading)",
                "requires_bitsandbytes": True,
                "supports_paging": True,
            },
            "paged_adamw8bit": {
                "name": "Paged AdamW 8bit",
                "description": "bitsandbytes PagedAdamW8bit (8-bit with CPU offloading)",
                "requires_bitsandbytes": True,
                "supports_paging": True,
            },
            "adafactor": {
                "name": "Adafactor",
                "description": "PyTorch Adafactor (adaptive learning rate, no momentum)",
                "requires_bitsandbytes": False,
                "supports_paging": False,
            },
            "lion8bit": {
                "name": "Lion 8bit",
                "description": "bitsandbytes Lion8bit (sign-based momentum, 8-bit)",
                "requires_bitsandbytes": True,
                "supports_paging": False,
            },
            "paged_lion8bit": {
                "name": "Paged Lion 8bit",
                "description": "bitsandbytes PagedLion8bit (sign-based momentum, 8-bit with CPU offloading)",
                "requires_bitsandbytes": True,
                "supports_paging": True,
            },
        }

        return optimizer_info.get(optimizer_type.lower(), {
            "name": f"Unknown ({optimizer_type})",
            "description": "Unknown optimizer",
            "memory_reduction": 0,
            "requires_bitsandbytes": False,
            "supports_paging": False,
        })
