#### CNNs build up features from edges and textures to full objects. How did this idea of progressive feature extraction influence the way you thought about refining your BBO strategy?

-   CNNs learn hierarchically: early layers detect simple patterns (edges, textures), while deeper layers combine these into complex object representations. A
    similar idea appears in Bayesian Optimization (BO) with Gaussian Processes (GPs) through **progressive refinement**.

    Initially, the GP has high uncertainty and models the objective function coarsely, analogous to CNNs learning basic features. The acquisition function then
    explores broadly to gather informative samples. As more evaluations are collected, the GP builds a richer representation of the objective landscape,
    identifying promising regions. BO gradually shifts from exploration to exploitation, focusing samples near likely optima. Thus, like CNNs moving from simple
    to complex abstractions, I also refined the BBO strategy from understanding broad global structure to precise local optimization, improving efficiency and
    accuracy over successive iterations.

#### LeNet and later CNNs redefined what is possible in computer vision. What parallels do you see between those breakthroughs and the incremental improvements you make in your BBO capstone project?

-   I see a clear parallel between the evolution of CNNs and Bayesian Optimization (BO) with Gaussian Processes (GPs): both achieved major advances through a
    series of incremental improvements rather than a single breakthrough.

    LeNet showed that hierarchical feature learning could outperform handcrafted features, while later CNN innovations improved accuracy, scalability, and
    efficiency. Similarly, BO started with the idea of modeling expensive objective functions using GPs. Over time, enhancements such as better kernels,
    acquisition functions (Expected Improvement, UCB, Thompson Sampling), and scalable GP methods made optimization more effective.

    To me, the key similarity is that both fields progressed by continually refining how they represent information, manage uncertainty, and use computational
    resources.

#### Training CNNs often involves balancing depth, computational costs and overfitting risks. Did you face similar trade-offs when choosing whether to explore widely or exploit promising regions in your queries?

-   Yes, abolutely. For each weekly submissions, we need to do a trade-off between exploration and exploitation.

    Personally, I relied upon a simple rule: compare the output of a given weekly query against the best output (maximum value) observed till the preceding
    week. If the latest output is lesser than the current maximum, we need more exploration to unearth hitherto unexplored regions (which may contain even
    higher values). Conversely, if the latest output is greater than the current maximum, we are possibly in a more promising region and need more exploitation
    to fine tune our search.

    However, the switch from exploration to exploitation may not necessarily be an one-time decision, since these black-box functions can have multiple maxima &
    minima. So, while we may try to fine-tune our search in potentially promising region, we may unknowingly get stuck in a local max/min and miss the global
    one - which is the broad objective of Bayesian Optimisation. Hence, if further improvement (that is, output values higher than the current maximum) is not
    detected within 2-or-3 iterations of exploitation, I revert back to exploration to explore other regions that may contain the globally optimum value. This
    way, we scan the function space thoroughly and give ourselves the most chance to discover the global optimum. And since we record each data-point searched
    and the corresponding output value, in the end, we can always know the best overall value that has been found.

#### Convolutions, pooling, activations and loss functions influence how CNNs learn from data. Which of these concepts helped you think differently about how your optimisation model learns from your accumulated data?

-   **Convolutions vs. Surrogate Model**: CNN components describe how information is transformed and distilled, while Bayesian Optimization (BO) is about how
    information is accumulated and used to decide where to learn next. Thinking across the two can give useful intuition. - A convolutional filter scans local
    regions and learns reusable patterns.
    -   Similarly, a BO surrogate model (often a Gaussian Process) processes observed samples and learns patterns in the objective landscape. Each new
        observation influences nearby regions in the input space through the covariance structure. The Gaussian Process kernel is somewhat analogous to a
        convolution filter because it determines how information propagates from observed points to unobserved ones.
-   **Pooling vs. Accumulation of evidence**: Pooling compresses many activations into a simpler representation while retaining important information. In BO,
    the surrogate's posterior mean and uncertainty summarize all previous observations. Instead of remembering every evaluation independently, BO forms a
    compact statistical summary. As such, GP posterior can be interpreted as a sophisticated form of pooling over all historical experiments:
    -   keeping the signal
    -   smoothing noise
    -   estimating confidence
-   **Activations vs. Acquisition functions**: Activations determine how intermediate representations drive later decisions in a network. In BO, the acquisition
    function (Expected Improvement, UCB, Probability of Improvement, etc.) transforms model predictions (mean & uncertainty) into action (next data-point to
    search). So:
    -   CNN activation: converts feature values into useful non-linear signals.
    -   BO acquisition: converts beliefs into search behaviour.
-   **Loss function vs. Objective function**: A CNN learns parameters to minimize loss. BO seeks inputs that optimize an objective function.

    The difference is that:

    -   CNNs observe many examples and directly optimize weights.
    -   BO treats evaluations as expensive and learns a model of the objective before deciding where to evaluate next.

    So the objective function in BO plays a role analogous to the loss surface in deep learning.

#### The interview with Andrea Dunbar highlighted the trade-offs of deploying CNNs in edge AI systems. How might reflecting on real-world deployment challenges help you decide how to benchmark success in your own BBO capstone project?

-   A useful takeaway from Andrea Dunbar's discussion is that success in deployment is often different from success in development. A CNN might achieve
    excellent accuracy in a lab, but if it is too slow, power-hungry, or unreliable on an edge device, it may still fail in practice.

    Below are some deployment considerations comparable between Edge CNN and Bayesian Optimisation.

    | Edge CNN Question                     | Bayesian Optimization Question                                   |
    | ------------------------------------- | ---------------------------------------------------------------- |
    | Can the model run on an edge device?  | Can the optimizer operate within a stipulated evaluation budget? |
    | What is the power cost?               | What is the computational and experimental cost?                 |
    | Is accuracy stable in the real world? | Is optimization performance stable under noise and uncertainty?  |
    | How fast is inference?                | How quickly does BO identify useful regions of the search space? |

    So reflecting on deployment challenges encourages us to define multi-dimensional success criteria. The "best" BO method may not be the one that reaches the
    absolute optimum after 1,000 evaluations; it may be the one that finds a near-optimal solution within the realistic budget, time, and uncertainty
    constraints of the real-world application.
