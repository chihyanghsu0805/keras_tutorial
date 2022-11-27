## Model Analysis

`Error Analysis` is used to understand which part of workflow results in most error and guide where to improve the workflow. Similarly, `Ceiling Analysis` is used to understand which part of workflow has the most potential improvement.

A common error is `Mislabeled Samples`. For big enough train set, it may not pay-off to correct all mislabeled samples. But mislabeled samples in dev and test set should always be corrected and ensure dev and test set ctill comes from same distribution. Additionally, examine the correct samples as well.

`Ablation Analysis` is used to understand removing which part of the model has the most impact.

## Other Concepts

-   `Transfer Learning` is a useful technique when two datasets A and B has same input (image, audio), and there are more data for A and not as much for B. Models trained on A can be used for `pre-training` for B and iterate some epochs for `fine-tuning` under the assumption that the lower level features may be helpful.

-   `Knowledge Distillation` is used for transferring knowledge from one large model to one small model. An example is `teacher student network`.

-   `Multi-Task Learning` is usefful when a set of tasks could share lower level features, for example obeject detection with multiple labels in image. The label distribution should be similar, and the dataset should be big enough.

-   `End-to-End learning` is helpful when each of the `subtasks are easier` and there are `more data for subtasks`.
