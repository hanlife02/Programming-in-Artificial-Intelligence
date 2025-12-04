# task1_optimizer Report

## Overview
`task1_optimizer.py` implements a NumPy training loop for a two-layer fully connected network that classifies MNIST digits. Torchvision is only used for downloading and normalizing the data; every forward pass, loss computation, and optimizer update is coded directly with NumPy so the optimization behavior is transparent.

## Data Handling (`parse_mnist`)
- Applies `ToTensor` and per-channel normalization before materializing the entire train and test splits through a single `DataLoader` pass.
- Flattens each 28x28 image to 784 features and casts images to `float32`, labels to `int64`.
- Returns `(X_tr, y_tr, X_te, y_te)` arrays that feed the training loop without any external dependencies.

## Network Structure (`set_structure`, `forward`)
- `set_structure` seeds the model as a 2-layer MLP: `W1/b1` map the input to a hidden layer with ReLU activation, and `W2/b2` project the hidden activations to class logits. Weights use variance-scaled Gaussian draws and biases start at zero.
- `forward` performs `hidden = relu(X @ W1 + b1)` and `logits = hidden @ W2 + b2`, returning the logits used by the softmax loss and accuracy computation.

## Loss and Metrics (`softmax_loss`, `loss_err`)
- `softmax_loss` is a numerically stable cross-entropy implementation that subtracts the row-wise max logit, exponentiates, and averages the negative log-likelihood.
- `loss_err` pairs the loss with 0/1 error computed from `argmax` predictions, letting the training loop report both values per epoch.

## Optimizers (`SGD_epoch`, `Adam_epoch`)
- `SGD_epoch` shuffles data indices, iterates over mini-batches, performs manual forward/backward passes (reusing the ReLU mask), and applies in-place SGD updates.
- `Adam_epoch` shares the same gradients but tracks first and second moments (`m`, `v`) per parameter. A persistent state dictionary keyed by the weight list id stores `m`, `v`, and timestep `t`; each mini-batch increments `t`, applies bias correction, and updates with `eps = 1e-8` for stability.

## Training Configurations and Results
All experiments below use hidden_dim=100, epochs=20, batch=256.

### Adam (`lr=0.005`, `beta1=0.9`, `beta2=0.999`)
| Epoch | Train Loss | Train Err | Test Loss | Test Err |
|-----:|-----------:|----------:|----------:|---------:|
| 0 | 0.11753 | 0.03478 | 0.12575 | 0.04110 |
| 1 | 0.08070 | 0.02407 | 0.10260 | 0.03320 |
| 2 | 0.06138 | 0.01737 | 0.08852 | 0.02880 |
| 3 | 0.04958 | 0.01352 | 0.07981 | 0.02550 |
| 4 | 0.04201 | 0.01133 | 0.07770 | 0.02550 |
| 5 | 0.04020 | 0.01172 | 0.07699 | 0.02430 |
| 6 | 0.03299 | 0.00823 | 0.07785 | 0.02410 |
| 7 | 0.02785 | 0.00737 | 0.07297 | 0.02390 |
| 8 | 0.02477 | 0.00590 | 0.07137 | 0.02170 |
| 9 | 0.02260 | 0.00508 | 0.07604 | 0.02470 |
| 10 | 0.01907 | 0.00400 | 0.07093 | 0.02270 |
| 11 | 0.01918 | 0.00443 | 0.07399 | 0.02170 |
| 12 | 0.01440 | 0.00240 | 0.07083 | 0.02120 |
| 13 | 0.01550 | 0.00315 | 0.07695 | 0.02370 |
| 14 | 0.01243 | 0.00207 | 0.07173 | 0.02200 |
| 15 | 0.01097 | 0.00165 | 0.07207 | 0.02150 |
| 16 | 0.00943 | 0.00115 | 0.07543 | 0.02240 |
| 17 | 0.00854 | 0.00105 | 0.07604 | 0.02210 |
| 18 | 0.00673 | 0.00062 | 0.07482 | 0.02110 |
| 19 | 0.00877 | 0.00183 | 0.07938 | 0.02170 |

### SGD (`lr=0.1`)
| Epoch | Train Loss | Train Err | Test Loss | Test Err |
|-----:|-----------:|----------:|----------:|---------:|
| 0 | 0.23550 | 0.06912 | 0.23386 | 0.06850 |
| 1 | 0.16344 | 0.04613 | 0.16921 | 0.05000 |
| 2 | 0.13548 | 0.03872 | 0.14516 | 0.04220 |
| 3 | 0.10617 | 0.02978 | 0.11967 | 0.03410 |
| 4 | 0.09394 | 0.02662 | 0.10988 | 0.03330 |
| 5 | 0.08753 | 0.02503 | 0.10703 | 0.03300 |
| 6 | 0.07229 | 0.02030 | 0.09412 | 0.02830 |
| 7 | 0.06785 | 0.01845 | 0.09129 | 0.02660 |
| 8 | 0.06239 | 0.01720 | 0.08922 | 0.02670 |
| 9 | 0.05737 | 0.01575 | 0.08703 | 0.02570 |
| 10 | 0.05567 | 0.01565 | 0.08817 | 0.02600 |
| 11 | 0.05038 | 0.01367 | 0.08404 | 0.02600 |
| 12 | 0.04847 | 0.01308 | 0.08650 | 0.02730 |
| 13 | 0.03916 | 0.00950 | 0.07903 | 0.02430 |
| 14 | 0.03632 | 0.00873 | 0.07641 | 0.02230 |
| 15 | 0.03597 | 0.00908 | 0.07753 | 0.02320 |
| 16 | 0.03126 | 0.00688 | 0.07507 | 0.02260 |
| 17 | 0.02944 | 0.00648 | 0.07445 | 0.02240 |
| 18 | 0.02810 | 0.00592 | 0.07605 | 0.02380 |
| 19 | 0.02609 | 0.00548 | 0.07356 | 0.02270 |

### Observations
- Adam converges faster in the early epochs and reaches sub-2% training error by epoch 12, but shows mild overfitting after epoch 10 as the test loss bottomed out near 0.071.
- SGD starts slower yet ends with comparable 2.2% test error while maintaining smoother loss curves, demonstrating that both optimizers can reach similar accuracy once tuned.
