# BEQ-BBR

Official implementation of **BEQ: Behaviour Evidence Querying with Asymmetric Learning for Bodily Behaviour Recognition**.

This repository contains our solution for the **MultiMediate Bodily Behaviour Recognition (BBR)** challenge. The method adapts a large vision-language model to multi-label bodily behaviour recognition and uses behaviour-specific queries to extract category-relevant evidence from video tokens.

## Method Overview

Our framework consists of three main parts:

1. **LVLM Backbone**
   Sampled video frames are processed by Qwen3-VL to extract visual-semantic token representations.

2. **Behaviour Evidence Querying (BEQ)**
   Each behaviour category has a semantic query, which cross-attends to LVLM video tokens and extracts category-specific evidence.

3. **Asymmetric Learning**
   Asymmetric Weighted Loss and class-balanced positive reweighting are used to handle the long-tailed multi-label distribution.

