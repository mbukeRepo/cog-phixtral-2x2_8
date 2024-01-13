# cog-phixtral-2x2_8

Cog wrapper for [phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8)

## Description

phixtral-4x2_8 is the first Mixure of Experts (MoE) made with four microsoft/phi-2 models, inspired by the mistralai/Mixtral-8x7B-v0.1 architecture. It performs better than each individual expert.

## Inferencing

```bash
cog predict -i prompt="Write a detailed analogy between mathematics and a lighthouse"
```
