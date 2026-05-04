# AI-ML Capstone Project

Source code of Capstone project from AI-ML professional course - Imperial College Executive Education.

## Section 1: Project Overview

This project is essentially a black-box optimisation (BBO) challenge where the goal is to find the maximum output of 8 unknown (black-box) functions. The exact
nature of these 8 functions is not known, however, a set of input data-points and their corresponding output values are provided per function as initial
data-set. The only way to learn further about the functions is by submitting 1 query point per function, to which the course administrators send back an output
value via the project portal. The output received can then be used to revise the maximum value so far per function, make an educated guess about the next query
point and repeat the process for a fixed number of times. The project presents an ideal use-case where **Bayesian Optimisation** algorithm can be applied to
discover the optimum solutions of largely unknown functions that are potentially expensive to evaluate and exhibit a high degree of uncertainty.

### Overall goal of the project

-   Find the maximum output of 8 unknown (black-box) functions over a defined search space. The functions are multi-dimensional, that is - they have multiple
    input features, from 2 all the way up to 8. However, in terms of output, each function produce a single value and it is this value that needs to be
    maximised.

### Relevance to real-world ML

-   There are many scenarios where the business objective, when modeled as a function, is expensive to evaluate, time-consuming and has a high degree of
    uncertainty. Bayesian Optimisation, based on a probabilistic model, is a powerful technique to find optimum solutions in such complex, expensive, or
    high-dimensional systems. Some of the well-known use-cases where Bayesian Optimisation can be very effective are:
    -   Hyperparameter Tuning
    -   Robotics
    -   Chemical Engineering
    -   Scientific Simulations

### Career relevance

-   AI & ML are no longer optional, nice-to-have, CV-attractive skills anymore; they are becoming mainstream by the day. Lack of fluency & familiarity in AI-ML
    techniques/frameworks/methods can quickly render an IT professional obsolete in the intense competitive domain of engineering & technology jobs. The
    Capstone project serves as a strong evidence of applicability of one of the widely known ML techniques - Bayesian Optimisation and demonstrates its
    strengths. The project showcases use of some of the prominent aspects of Bayesian Optimisation - such as; a surrogate function (often a Gaussian Process),
    mean prediction, uncertainty estimate, acquisition function (PI, EI, UCB etc.) and balancing exploration vs. exploitation. These topics will increasingly
    become essential for software engineers, solution architects & technical leads to design & develop next generation intelligent systems with ever more
    growing capabilities.

## Section 2: Inputs and Outputs

Since the exact nature of the functions are hidden, the only way to learn about the functions is to submit a query point and receive the corresponding output
value in return. Each query point contains a number of fractions. The number of fractions for a given function depends upon the dimensions of the input
features. Thus, for a function with 2-D input features, a query point should contain 2 fractions, 3 fractions for 3-D input features and all the way up to 8-D
input features.

-   Inputs
    -   Initial: A data-set containing a several input data-points per function. The size of this data-set varies from function to function.
    *   Weekly: 1 data-point per function. This data-point represents the co-ordinates where the (black-box) function should be evaluated next. The choice of
        this data-point can be guided by personal intuition, heuristics or the Bayesian Optimisation program that can deduce the next best point to query,
        depending upon exploration vs. exploitation trade-off. A data-point contains a number of fractions, this number is equal to input features dimensions of
        the function in question. Each fraction falls within the range: (0.0 - 1.0) and has precision of up to 6 decimal places. As an example, a data-point for
        a funtion with 4-D input features may be arranged this way - `0.372186-0.504686-0.319187-0.073657`

*   Outputs
    -   Initial: A data-set containing the output values corresponding to the input data-points per function.
    -   Weekly: 1 output value per function. This value represents the output computed by the (black-box) function against the input data-point submitted via
        the project portal for the same week. Unlike input data-point co-ordinates, the output is a single value and does not have bounds. It can also be
        negative.

## Section 3: Challenge Objectives

-   Primary objective: For each (black-box) function, find the maximum value of the output within the search space (within the range 0.0 - 1.0 for each input
    feature) as well as the co-ordinates that produced this value. If the function, by definition, produces negative numbers, then find the output that is
    closest to 0.

*   Constraints: Since only a single data-point can be queried per week, the number of evaluations per function is quite limited; only about 10 - 12. Similar to
    real-world use-cases this strictly fixed number of queries represent a limit evaluation budget. Also, since the actual nature of all these functions are
    unknown, there is no knowledge of the curvature, gradients, smoothness or modality which may guide the discovery of the maximum outout and the corresponding
    input data-point. Finally, the output value generated against a given input query point may contain noise which can introduce randomness.

## Section 4: Technical Approach

### Core Strategy

Construct a set of programs based on Bayesian Optimisation algorithm that can substitute the actual (black-box) function with a proxy, that is much simpler,
economical and light-weight than the actual. This proxy is known as a **surrogate model** (typically a **Gaussian Process**). Then, the surrogate model is
optimised, through an iterative process, until the optimum output (in this project, the maximum) reaches a plateau or the iteration count reaches a certain
limit (in this project the maximum no. of queries allowed).

<P></P>

A `GaussianProcessRegressor` combines the prior probability and the likelihood function based on the weekly data-points. It can then make probabilistic
predictions about the nature of the function (which is actually hidden). The prediction consists of the mean output and the standard deviation (which represents
uncertainty) at every point throughout the search space. The predicted mean and the standard deviation is then combined to derive an **acquisition function**
which can then hint at the next data-point to evaluate. The acquisition function can be adjusted to favour exploration or exploitation via a scaling factor
(`z_score`). Once the next query point for all 8 functions have been computed (which may be selected intuitively, bypassing the acquisition function altogether)
they are submitted via the project portal. When the evaluated outputs against these query points are sent back, they are added to the weekly data-set (which
grows cumulatively) and the optimisation process moves to the next iteration.

### Weekly Updates

-   Week-1: In early stages of Bayesian optimisation, exploration should be preferred over exploitation as most the possible outputs of the true, objective
    function are not known. In such cases, the query can be formed based on data-points at random or some average/mid-way value. Here, the latter has been
    chosen across all 8 objective function.

*   Week-2: Depending upon the output value of Week-1 returned for any given function, small changes can be made to certain hyper-parameters to move towards
    higher output values. That is, hyperparameters (e.g., **confidence interval** or `z-score`) can be adjusted that would direct the search to hitherto
    unexplored regions (exploration) or in the vicinity of already known high values (exploitation).

*   Week-3: Graphical visualization is the most important capability added this week to the query strategy; this wasn't there during the first 2 weeks.
    Currently, a hybrid approach is being followed where adjustments are made to a few hyper-parameters (such as; **confidence interval** or `z-score`, **noise
    estimate**, `length_scale`, **smoothness conefficient** or `nu`) as well as observing the nature of the functions graphically. A **Gaussian Process
    Regression (GPR)** graph helps to understand the broad trends of a function against its inputs and a **Scatter Chart** shows the data-points known so far in
    the same function space. The graphical representation serves as a sort of validation of the hyper-parameter adjustments hinted by the GPR model. The
    **Radial Basis Function (RBF)** kernel is used as the default choice, but sometimes other kernels (such as; **Matern**, **RationalQuadratic**,
    **ConstantKernel** etc.) are tried too, to check if the predicted query-points align more closely with GPR graph.

<P></P>

### Suitability of SVM

This BBO challenge is a _regression_ problem, not a _classification_ one. The **Support Vector Machine (SVM)** is a classification technique, not a regression
one. Therefore, SVM is not the appropriate method to discover the optimum value of a quantity (in this case, maximum outout). However, SVM can be a good choice
if we wish to segregate regions on the basis of their potential to produce high or low value outputs. A soft-margin SVM is a better choice than hard-margin one
due to its ability to accommodate outliers (e.g., a high output from a primarily low-performance region). In order to detect the best separation hyper-plane, a
**polynomial** or **RBF** kernel should be used as they can adjust to non-linear relationships as well as produce smooth surfaces.

### Strength of the Solution

Bayesian Optimisation is a powerful algorithm to find optimum results where the problem domain has high degree of uncertainty, trials/evaluations are expensive
and time-consuming. Because it produces probabilistic results, it can refine these probabilities upon addition of fresh data-points and converge towards the
global optimum in an iterative manner. The solution for this BBO challenge has been designed as a configurable framework that can run Bayesian Optimisation
every week against already accumulated and new data and progressively refine its predictions. It also has facilities to depict the functions visually which
makes it convenient to compare the behaviour of the functions against the predictions made by the `GaussianProcessRegressor`. And because these programs have
been designed as a framework, there is hardly any requirement of code change from week-to-week, making the entire optimisation process easily repeatable.
