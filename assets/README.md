# README images

`images/overview.png` is the author-provided `overviewing.png`, reproduced byte-for-byte without cropping, resizing or redrawing. It illustrates the text-conditioned point-cloud diffusion architecture with token cross-attention, stop-word reweighting and FiLM conditioning.

The figure labels its text encoder as CLIP ViT-L/14. The current code default remains `openai/clip-vit-base-patch32`; consult the checkpoint configuration for the encoder used in a particular experiment.

This replaces the earlier upstream baseline teaser. Verified qualitative result figures should include the originating prompt and checkpoint or paper-figure provenance.
