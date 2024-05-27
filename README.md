# Weakly-supervised-PU-Learning-from-Imbalanced-Data

This repository presents a novel approach to binary classification problems where negative data is diverse and challenging to label completely.In many real-world problems insufficient labeled data and class imbalance usually coincide and achieving a high detection rate for the minority class is vital. Our method, PU+PN, is designed to address these challenges by reducing the False Negative Rate (FNR) for the minority class, which is crucial when negative data is absent and unlabeled data are treated as negative.

## Experimentation on MNIST and CIFAR10

We conducted extensive experiments on the MNIST and CIFAR10 datasets, which are well-suited for this kind of binary classification task as both datasets contain 10 classes. In our experiments, one class is considered positive, and the remaining nine are treated as negative. The `neg_ps` parameter indicates which negative classes and how much contribute to the partially-observed negative data used for training.

### Scenario 1: Varying Labeled Negative Instances

The first scenario explores the impact of varying the number of labeled negative instances on performance metrics. By incrementally adding labeled negative samples from the only observed negative class into the training set, we analyze how the ratio of labeled training data affects the model's performance.

### Scenario 2: Different Numbers of Known Negative Classes

The second scenario evaluates performance metrics across different numbers of observed negative classes. Here, we maintain a constant overall ratio of labeled training examples at 0.05 while increasing the number of observed negative classes in the training data(using neg_ps)

### Hyperparameters

•  `pi`: The ratio of labeled positive instances.

•  `rho`: The ratio of labeled negative instances.


### Training Modes

•  `Oversampling`: Indicates whether data augmentation is used to balance the classes during training.


### Methods

The parameters below determine which method (the main method or baselines) must be run:
•  `iwpn`: False

•  `pu_then_pn`: False

•  `PUplusPN`: True

•  `PN_base`: False

•  `pu`: False

•  `pnu`: False

### Model

In this implementation, a 13-layer CNN is used for the classification task


Our approach aims to provide a robust solution to the PU classification problem, ensuring high detection rates for the minority class while dealing with the challenges of unlabeled and imbalanced data.

