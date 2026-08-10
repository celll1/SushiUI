# FBCache acceptance and MiniMax-H3 safeguards

FBCache is an opt-in approximation. Same-seed LPIPS and SSIM measure distance
from the uncached denoising trajectory; they do not by themselves establish
that a generated video is unusable. Evaluation therefore has two stages:

1. Hard failure: reject non-finite output, black/frozen spans, subject loss or
   duplication, abrupt composition reset, topology collapse, severe temporal
   warp/flicker, conditioning-boundary violations, or broken/missing audio.
2. Blind comparison: rate prompt and camera adherence, subject/background
   consistency, motion continuity, local anatomy, audio continuity and overall
   usability. Ship a preset only when its speed/quality pair is useful.

## MiniMax-H3

The H3 transformer packs text, reference/keyframe video, target video and audio
into one self-attention sequence. Its cache therefore stores the full packed
hidden-state residual after all blocks. On a hit, the whole residual is reused;
video and audio can never make separate skip decisions.

The decision signal uses only generated target-video rows after block 0.
Alongside the global relative-L1 comparison, those rows are reshaped by the
layout's `rows_per_frame` and the largest per-latent-frame relative-L1 change is
used as a guard. Reference and keyframe rows are excluded because they are
fixed conditioning, not evidence that the generated trajectory is stationary.
The temporal guard and hit-chain cap are a clean-room implementation informed
by the MIT-licensed [MiniMax-H3 FirstBlockCache project](https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache).

The public `fbcache_threshold` and `fbcache_warmup_steps` remain unchanged. H3
adds fixed internal limits: no more than two consecutive hits and one mandatory
real evaluation at the tail. FBCache is disabled when Block Swap or Spectrum is
active. The disabled default still calls the unwrapped stock transformer.

Use `0.08` as the conservative H3 starting threshold. The shared `0.12` default
is intentionally not rewritten behind the user's back, but earlier uncapped,
unguarded measurements show that it is a substantially more aggressive H3
setting. The guarded mechanism still requires real-generation A/B coverage of
text-to-video, reference/keyframe conditioning, temporal inpaint/outpaint and
audio before any stronger quality claim.

One post-implementation txt2vid smoke at 640x384x124, 20 steps, W4A8 + Flash
Attention and threshold `0.08` reduced denoise time from 62.689 s to 46.631 s
(25.6%). Across all 124 decoded frames, exact-reference SSIM was 0.9912, PSNR
39.33 dB and LPIPS-Alex 0.0175 mean / 0.0240 max. Both outputs retained H.264
video and 32 kHz stereo AAC audio; visual review found no black frame, freeze,
subject loss or structural collapse. This validates the basic path, not the
remaining conditioned-workflow matrix above.
