#### What is the main technical justification for your current BBO approach? Which aspect of prior research or established methods supports your choice?

-   For the BBO challenge, I used Bayesian Optimisation with Gaussian Processes as surrogate model.

    Bayesian Optimization commonly uses **Gaussian Processes (GPs)** because GPs provide a flexible probabilistic surrogate model that estimates both the
    **expected value** of an unknown objective function and the **uncertainty** of that estimate. This uncertainty quantification is crucial for efficiently
    balancing **exploration** (sampling uncertain regions) and **exploitation** (sampling promising regions) through acquisition functions such as Expected
    Improvement or Upper Confidence Bound.

    The main technical justification comes from established work in **spatial statistics (kriging)** and Bayesian nonparametric modeling, where GPs have long
    been used for interpolation of expensive black-box functions. Foundational research by Jones et al. (1998) on Efficient Global Optimization and later
    studies demonstrated that GP-based surrogates achieve sample-efficient optimization, making them particularly suitable when function evaluations are costly.

#### Which academic papers have you used to guide your design? Which ideas or techniques from the literature are most relevant, and how do they strengthen your project?

-   In terms of academic papers, modules 11 (Naïve Bayes theorem) and module 12 (Bayesian Optimisation) were my primary source of knowledge, although I did
    explore couple of online books (Bayesian Optimization in Action, Quan Nguyen; Bayesian Optimization: Theory and Practice Using Python, Peng Liu) to
    understand the optimisation process and related concepts (e.g., surrogate model, acquisition function, covariance matrix) at a deeper level.

    Bayesian Optimization using Gaussian Processes is well suited to **black-box optimization** because it builds a probabilistic **surrogate model** of an
    unknown, expensive-to-evaluate function. A Gaussian Process provides both a predicted function value and an estimate of uncertainty, allowing the optimizer
    to make informed decisions about where to evaluate next.

    This uncertainty information is used by **acquisition functions** such as Expected Improvement (EI) and Upper Confidence Bound (UCB) to balance
    **exploration** of uncertain regions with **exploitation** of promising areas. As a result, the method can locate near-optimal solutions using relatively
    few costly function evaluations.

    The approach is supported by established work in **kriging (spatial statistics)** and Bayesian surrogate modeling, where Gaussian Processes have been
    successfully used to model and optimize unknown functions from limited data. This strong theoretical foundation and high sample efficiency make GP-based
    Bayesian Optimization a standard method for black-box optimization problems.

#### Which third-party libraries or frameworks (e.g. PyTorch, TensorFlow, scikit-learn) are central to your approach? Why were these the right choices compared with possible alternatives?

-   My approach is based on the scikit-learn library and primarily on the sklearn.gaussian_process & sklearn.gaussian_process.kernels packages. The scikit-learn
    library contains classes like <code>GaussianProcessRegressor</code> and various kernel implementations such as <code>RBF</code>, <code>Matern</code>,
    <code>RationalQuadratic</code> etc. These library classes are very effective to implement an iterative Bayesian Optimisation process and they also offer
    customisation options through various hyper-parameters like <code>alpha</code> (noise estimate), <code>n_restarts_optimizer</code>,
    <code>normalize_y</code>, <code>length_scale</code>, <code>nu</code> (smoothness) etc.
-   Scikit-learn is often preferred for Bayesian Optimization exercises because it is simpler, faster, and more stable than PyTorch or TensorFlow. Most
    scikit-learn models train quickly, use a standardized fit/predict API, and have relatively small hyperparameter spaces, making them easy to optimize. They
    are also more deterministic, reducing noise in objective evaluations and improving optimization efficiency.

    In contrast, PyTorch and TensorFlow require custom training loops, involve larger hyperparameter spaces, and introduce stochasticity from random
    initialization and SGD, which can make Bayesian Optimization more challenging.

    For learning Bayesian Optimization concepts and tuning classical ML models, scikit-learn is usually the most practical choice. For deep learning
    hyperparameter tuning, PyTorch or TensorFlow are more appropriate.

#### How do you plan to document and present these justifications in your GitHub repository so that peers, facilitators and future employers can clearly understand your reasoning?

-   I am storing all artifacts related to this <b>Black-box Optimisation</b> project in a designated GitHub repository. The artifacts include the following -
    <ol>
        <li>
            All source code, written in Python.
            <ul>
                <li>
                    Including weekly Python programs for each function, as well as Python modules that can be reused across all functions and all weeks.
                </li>
            </ul>
        </li>
        <li>Weekly input query points for all 8 functions.</li>
        <li>Weekly output values for all 8 functions.</li>
        <li>Elaboration/explanation of weekly updates to the search strategy as required by capstone component assignments.</li>
        <li>
            A MS-Excel spreadsheet that lists all query points (starting from the initial sets and then on weekly basis), their corresponding output values, the GP
            regressor & kernel hyper-parameters for each function. The spreadsheet serves as a single place tracker on how the convergence to the optimum output
            value and its corresponding input data-point is progressing week-by-week.
        </li>
        <li>
            A README.md file that provides a project overview, its overall goals, relevance to real-life AI-ML use cases, relevance to career paths supercharged by
            AI-ML mechanisms, a description of the input & output data formats and a summary of the technical implementation.
        </li>
    </ol>
-   The combination of the source code, the tracker spreadsheet and the readme file provides a neat & elaborate description of the project's objective,
    constraints and solution strategy that can be showcased to facilitators and employers.

#### Looking ahead, what additional sources (research, benchmarks, software) might you consult to continue refining your strategy?

-   In the current exercise, I have used the <code>GaussianProcessRegressor</code> class to run surrogate models for each of the 8 "black-box" functions. I have
    switched between a few kernels (RBF, Matern, RationalQuadratic etc.) that are pre-packaged inside scikit-learn and tweaked their hyper-parameters like
    <code>length_scale</code>, <code>nu</code> (smoothness), to locate favourable query points. However, I have not tried customising the <code>optimizer</code>
    or write a custom kernel or combine multiple kernels in this exercise. The search process can be fine-tuned (efficieny, accuracy etc.) to a greater extent
    by customization to these components. I look forward to explore these customization avenues for similar projects in future.
-   Moreover, from a few additional resources outside of this course, I have come to know about other frameworks that may offer alternative ways of building GP
    models. One such framework is <b>GPyTorch</b>, an object-oriented extension of <b>PyTorch</b>, designed specifically for scalable, flexible Gaussian Process
    modelling, supporting advanced kernels, variational inference, GPU acceleration, and large datasets. For Bayesian Optimisation, GPyTorch (often used with
    BoTorch) offers greater customisation and state-of-the-art performance than scikit-learn. This is another area which I would like to explore for projects
    similar to this one.
