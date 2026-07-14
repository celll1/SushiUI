# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone, framework-free port of NVIDIA PiD (Pixel Diffusion Decoder), inference-only.

Ported from github.com/nv-tlabs/PiD (Apache-2.0). This package strips the original
`imaginaire`/hydra/lazy-config training framework and vendors only the code paths
needed to run the distilled 4-step SDXL student decoder:

    caption_embs -> pixel-space noise[B,3,H,W] -> PidInferenceModel (SDE sampler,
    velocity prediction, 4 steps) with the raw SDXL latent injected via LQProjection2D
    (ControlNet-style gated injection) -> [B,3,H,W] image in [-1, 1].

See `backend/core/models/pid/loader.py` for the SushiUI-facing entry point.

Citation:
    NVIDIA PiD: Pixel Diffusion Decoder (github.com/nv-tlabs/PiD), Apache-2.0.

Weights (not vendored here) are NSCLv1 (non-commercial) licensed separately.
"""
