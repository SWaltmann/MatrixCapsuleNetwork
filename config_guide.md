## Dataset Name

The `dataset_name` key specifies which dataset to use for training and evaluation.

Currently supported:
- `"smallnorb"`: The Small NORB dataset of 3D toy objects (5 categories × 10 instances × 9720 images).

This key is reserved for future support of additional datasets. Use `"smallnorb"` unless otherwise extended.


## Validation Split Strategy

The `val_split_strategy` key controls how the validation set is created from the training data.

Options:
- `"random"`: Randomly samples validation data across all instances.
- `"loio"`: Short for 'leave one instance out'. Leaves out one full instance from the dataset to mimic the test set setup (unseen subjects).

This affects generalization behavior and better reflects final evaluation if `"leave_one_instance_out"` is used.


## Batch Size

The `batch_size` key sets the number of samples processed in one forward/backward pass.


## Validation Fraction (`val_fraction`)

Defines the fraction of training samples to use for validation when `val_split_strategy` is set to `"random"`.

- Must be a **float between 0 and 1** (e.g., `0.1` for 10%).
- **Ignored** if `val_split_strategy` is set to `"loio"`.


## Model Size (`model_size`)

Specifies which version of the Capsule Network architecture to use, as described in the *Matrix Capsules with EM Routing* paper by Hinton et al.

Options:
- `"full"`: The full version of the network, which is the primary model used throughout the paper. It includes more capsule channels and routing units, resulting in higher parameter count and typically better performance.
- `"small"`: A reduced version of the network described in Section 5 of the paper. It uses fewer parameters to demonstrate the model's efficiency and generalization with a lightweight configuration.

This setting adjusts the number of capsules and other architectural details while keeping the overall structure of the model (e.g., EM routing) consistent.
