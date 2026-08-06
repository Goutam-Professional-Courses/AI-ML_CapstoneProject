#### How did the ideas of hierarchical feature learning influence the way you thought about structuring or refining your optimisation strategy this round?

-   Hierarchical feature learning assumes that useful representations emerge at multiple levels of abstraction:

    -   Low levels capture simple local patterns.
    -   Middle levels capture reusable structures.
    -   High levels capture task-relevant concepts.

    A Bayesian optimiser faces a similar challenge: it must build a surrogate model of an expensive objective and decide where to sample next. The quality of
    its representation of the search space heavily affects performance. Hierarchical feature learning teaches us that not all variation occurs on the same
    scale. For BO this motivates surrogate models that capture:

    -   coarse global structure
    -   medium-scale trends
    -   fine local corrections

#### You saw how breakthroughs such as AlexNet and ImageNet classification reshaped expectations in AI. What parallels do you see between those leaps in performance and the incremental improvements you make in your capstone submissions?

-   AlexNet’s ImageNet success was a dramatic step change: it showed that a new combination of model architecture, data, and compute could vastly outperform
    existing approaches, reshaping expectations for AI.

    The parallel in Bayesian Optimisation (BO) with Gaussian Processes (GPs) is that performance gains also come from better modelling assumptions and more
    effective use of information. GPs encode beliefs about smoothness and uncertainty, while advances in BO improve how new evaluations are selected, increasing
    sample efficiency.

    However, the key difference is that GP-based BO has evolved mostly through **incremental improvements**—better kernels, acquisition functions, and
    scalability techniques—rather than a single breakthrough comparable to AlexNet. Whereas AlexNet represented a paradigm shift, GP-BO progress has been
    cumulative, steadily extending the range, robustness, and efficiency of optimisation methods.

    In short: both fields benefit from better inductive biases and computational tools, but AlexNet was a sudden leap, while GP-based BO has advanced through
    continuous refinement.

#### When training neural networks, people often weigh trade-offs between depth, complexity and training efficiency. Did you encounter similar trade-offs in deciding whether to explore widely or exploit known promising regions in your queries?

-   Yes. The exploration–exploitation trade-off in Bayesian Optimisation is closely analogous to the depth–complexity–efficiency trade-offs in neural networks.
    Exploration evaluates uncertain regions of the search space to gather new information. This is like increasing model capacity or depth to discover richer
    representations, even though training becomes more expensive. Exploitation focuses evaluations on regions already predicted to perform well. This is like
    simplifying a model or optimizing training efficiency to extract value from what is already known.
-   In both cases, pushing too far in one direction creates problems:
    -   Neural Networks:
        -   Too much complexity → slow training, overfitting risk
        -   Too much simplicity → underfitting
    -   Bayesian Optimisation
        -   Too much exploration → wasted evaluations
        -   Too much exploitation → local optima, missed opportunities
-   Personally, I relied upon a simple rule: compare the output of a given weekly query against the best output (maximum value) observed till the preceding
    week. If the latest output is lesser than the current maximum, we need more exploration to unearth hitherto unexplored regions (which may contain even
    higher values). Conversely, if the latest output is greater than the current maximum, we are possibly close to the best overall output and need more
    exploitation to fine tune our search for the optimum value.

#### Reflecting on the building blocks of neural networks (inputs, activations, loss, gradients, weight updates), which of these concepts helped you think differently about how your model learns from the data you’ve accumulated so far?

-   Several neural-network concepts provide useful intuition for how a Gaussian Process (GP) learns, even though the mechanics are different:

    -   **Inputs** → In both cases, the model learns a relationship between inputs and outputs. For a GP, the kernel defines how similar inputs influence each
        other.
    -   **Activations** → There is no direct equivalent, but the **kernel function** plays a similar role by shaping the model's expressiveness and the patterns
        it can represent.
    -   **Loss** → Instead of minimizing a training loss like cross-entropy or MSE, a GP typically fits its hyperparameters by maximizing the **marginal
        likelihood** (or minimizing its negative). This measures how well the observed data fit the model assumptions.
    -   **Gradients** → Gradients are still important. They are used to optimize kernel hyperparameters such as length scales and noise levels.
    -   **Weight updates** → This is where the analogy breaks down. Neural networks learn by iteratively updating weights. A GP has no hidden-layer weights; it
        "learns" by updating its posterior distribution after observing data and by adjusting a relatively small set of hyperparameters.

-   A useful mental model is: **a neural network learns by changing parameters, whereas a Gaussian Process learns by updating beliefs about functions.**

#### Module 16 also introduced PyTorch and TensorFlow as different frameworks for building and scaling models. If you were to frame your current optimisation approach in terms of a ‘framework’, would it be closer to rapid prototyping and flexibility or to structured, production-ready design? Why?

-   My optimisation approach is already organised somewhat similar to a _"framework"_. I have designed it in such a way that most of the code is contained
    within reusable Python modules & functions, requiring very little coding work on a week-by-week basis. The code that is specific to a given week, is
    contained within a directory that is named after the week (Week-2, Week-3, Week-4 etc.). The directory contains 1 Jupyter notebook and couple of Python
    files (for graphical representation) for each of the 8 black-box functions. The weekly directory also contains the input data-points and the output values
    for all 8 functions up until that week. The reusable Python modules & functions, however, are stored in the parent directory of the weekly ones. These
    reusable components are invoked from the weekly Jupyter notebooks, the highest output value till the current week calculated and the data-point for the next
    query selected. The progress of the optimisation process in every week, can be visualized by running the other, ancillary Python programs.
-   I have designed the BBO solution in this structure so as to maintain high consistency and uniformity as we advance through the weeks and keep code
    repetition to a minimum. Although we still have 3 source code files (1 Jupyter notebook and couple of supplementary Python files) per function per week,
    they are almost identical to their equivalent counterparts across other weeks, with the only real difference being the week & function numbers and
    occasionally the kernel passed to the GP regressor. The structure of the overall solution can further be improved through object-oriented Python that offers
    advanced techniques like abstraction, polymorphism & type hierarchy - techniques that empower construction of highly modular & extensible software.
-   Given the design objectives and the structure of the codebase, the solution is closer to a highly consistent, modular, easily verifiable production-ready
    framework, with room for further extensibility and customisation through object-oriented principles.

#### In the guest interview, Giovanni Liotta discussed industry applications of deep learning in sport. How might reflecting on real-world deep learning use cases inform the way you benchmark success in your own capstone challenge?

-   Sports analytics shows that success is measured by better decisions and outcomes, not just predictive accuracy. Deep learning is used for athlete
    monitoring, motion tracking, injury-risk prediction, and tactical analysis, where practical impact matters most. Sports applications suggest that Bayesian
    Optimisation should be benchmarked on four dimensions:
    -   Efficiency — how quickly good solutions are found.
    -   Impact — whether optimisation improves the real-world outcome of interest.
    -   Robustness — whether performance holds under noise and uncertainty.
    -   Trade-off Management — how well competing objectives and constraints are handled.
