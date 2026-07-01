"""BEQ: Behaviour Evidence Querying for Bodily Behaviour Recognition.

This package implements the LVLM adaptation framework described in the paper
"BEQ: Behaviour Evidence Querying with Long-tail Aware Asymmetric Learning for
Bodily Behaviour Recognition". It provides:

* ``LVLMBodilyClassifier`` (``beq.modeling``) -- the global-pooling LVLM-LoRA
  baseline.
* ``BEQClassifier`` (``beq.decoder``) -- the Behaviour Evidence Querying decoder
  that replaces global pooling with category-conditioned evidence retrieval.
* Long-tail aware asymmetric learning (LTAL) loss components (``beq.losses``).
"""
