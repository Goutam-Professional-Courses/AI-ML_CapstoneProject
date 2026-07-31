### Hyperparameter Effects
Neural network hyperparameters have a significant impact on model convergence, training stability, and overall performance. One of the most influential hyperparameters is the learning rate. When set too high, the optimisation process can become unstable or diverge; when set too low, convergence is slow and the model may become trapped in poor local minima. Batch size also strongly affects behaviour: smaller batch sizes introduce stochasticity that can improve generalisation but reduce stability, while larger batch sizes lead to smoother, more stable training but may converge to less generalisable solutions.
<p></p>
Architectural hyperparameters such as the number of layers and number of neurons per layer control the capacity of the model. Increasing capacity can improve performance on complex tasks, but it also raises the risk of overfitting and training difficulties. Regularisation-related hyperparameters, such as dropout rate and weight decay, help mitigate overfitting and improve robustness, although excessive regularisation can result in underfitting and reduced accuracy.

### Discrete vs Continuous Hyperparameters
#### Discrete hyperparameters include:
-   Number of layers (network depth)
-   Number of neurons per layer (network width)
-   Activation function (e.g. ReLU, tanh, sigmoid)
-   Optimiser choice (e.g. SGD, Adam, RMSprop)
-   Batch size (when treated as categorical)
-   Use of batch normalisation (on/off)
-   Weight initialisation method
-   Learning rate schedule type

#### Continuous hyperparameters include:
-   Learning rate
-   Momentum coefficient
-   Weight decay / L2 regularisation strength
-   Dropout rate
-   Adam optimiser parameters (β₁, β₂, ε)
-   Label smoothing factor
-   Gradient clipping threshold

The distinction between discrete and continuous hyperparameters influences how they are tuned. Discrete hyperparameters often require broader exploration strategies such as grid search, random search, or evolutionary algorithms. Continuous hyperparameters, by contrast, vary smoothly and are well suited to optimisation methods such as Bayesian optimisation or other gradient free continuous search techniques. In practice, discrete architectural choices are often selected first, followed by fine tuning of continuous parameters.

### Application to Black Box Optimisation
When using a neural network as a surrogate model in a black box optimisation (BBO) setting, hyperparameter understanding is crucial. A poorly tuned surrogate can produce noisy or biased predictions, leading the optimiser towards suboptimal regions of the search space. As a result, I would prioritise hyperparameters that encourage stable training and good generalisation rather than maximum expressiveness.
<p></p>
Furthermore, black box optimisation methods can be applied directly to neural network hyperparameter tuning itself. Since validation performance can be treated as a black box objective, approaches such as Bayesian optimisation or evolutionary strategies can systematically improve network performance. This creates a consistent framework in which BBO techniques enhance both the surrogate model and the underlying optimisation task.
