# Diffusers provenance

The files in this directory derive from Hugging Face Diffusers commit
`6256aa7` and retain their Apache License 2.0 headers.

Local changes are limited to imports that bind the components to SushiUI's
installed Diffusers package and to integration fixes recorded in git history.

Caption encoding limits unused vocabulary logits to one token and retains the scoped final-RMSNorm bypass required by the released DiT's conditioning.
