## Pruning Pathways: A Comparative Analysis of Neural Network Optimization Techniques

### *Batch running command*

#### Each project has unique arguments or hyperparameters for training targets; please look at the main.py of the relevant section.

```bash
python3 main.py resnet18  cifar10 cifar100 fashionmnist
python3 main.py vgg16  cifar10 cifar100 fashionmnist
python3 main.py mobilenet_v3_large  cifar10 cifar100 fashionmnist
python3 main.py resnet50   imagenet
python3 main.py vgg16   imagenet
python3 main.py mobilenet_v3_large   imagenet
```

### *Summarize to compare various prune methods*

| **Algorithm**   | **Pruning Type**              | **Key Methodology**             | **Strengths**                    | **Weaknesses**                    | **Target Sparsity Handling**                                                          | **Applicability**               |
| --------------------- | ----------------------------------- | ------------------------------------- | -------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------- |
| NetworkPruner         | Unstructured                        | Magnitude-based with retraining       | Simple, effective, retrainable         | No structural pruning, retraining-heavy | User-defined (e.g., 0.5), threshold                                                         | General-purpose, retraining available |
| GrowingRegularization | Structured/Unstructured             | Growing regularization + pruning      | Flexible, large model support          | Complex tuning, static pruning          | Prune_ratio (e.g., 0.5), final step                                                         | Gradual sparsity induction            |
| LearnableThreshold    | Unstructured                        | Learnable soft thresholding           | Adaptive, training-integrated          | Tuning complexity, inexact sparsity     | Progressive (e.g., 0.7), stops                                                              | Training from scratch                 |
| LossModelPruner       | Unstructured                        | Taylor/magnitude/random importance    | Loss-sensitive, flexible criteria      | Resource-intensive (Taylor), uneven     | Global target (e.g., 0.2), threshold                                                        | Fine-tuning pre-trained models        |
| HybridPruner          | Structured + Unstructured           | Sensitivity + weight pruning          | Accuracy-balanced, dataset-adaptive    | Expensive sensitivity test, tuning      | Implicit via sensitivity (e.g., 0.2)                                                        | Deployment with high accuracy         |
| StructuredPruning     | Structured                          | L1 norm (channel/kernel/intra-kernel) | Versatile granularity, robust (PF)     | Particle filter overhead, niche use     | Explicit ratio (e.g., 0.5)                                                                  | Structured CNN pruning                |
| SNIP                  | Unstructured                        | Gradient-based single-shot pruning    | Fast, pre-training                     | Data-dependent, over-pruning risk       | Direct ratio (e.g., 0.5), threshold                                                         | Sparse initialization                 |
| WeightToNodePruning   | Unstructured                        | Probabilistic + forced sparsity       | Exact sparsity, node-aware             | Less interpretable, shape issues        | Strict ratio (e.g., 0.5)                                                                    | Precise sparsity research             |
| L1StructuredPruning   | Structured                          | L1 norm-based channel/kernel pruning  | Simple, minimal functionality          | Limited scope, no adaptation            | Fixed ratio (e.g., 0.5) per layer                                                           | Basic CNN structured pruning          |
| ADMMPruner            | Unstructured                        | ADMM with progressive sparsity        | Robust, accuracy-preserving            | Computation-heavy, tuning-intensive     | Scheduled target (e.g., 0.5)                                                                | Advanced high-accuracy pruning        |
| **DeepLSP**     | **Structured + Unstructured** | Sensitivity + latency-aware pruning   | Balances size/latency, adaptive ratios | Library dependency, dummy data reliance | **Structured: keep_ratio (e.g., 0.5); Unstructured: amount (e.g., 0.05) or adaptive** | **Deployment (edge devices)**   |
