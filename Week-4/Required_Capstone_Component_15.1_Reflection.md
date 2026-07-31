#### In your function evaluations, which inputs seemed to act like support vectors – points near a decision boundary or region of rapid change How might recognising them guide your next query?

-   Function 1: Between 0.5 - 0.73 for both input features, as the output goes from +ve to -ve and back again to +ve.
-   Function 2:
    -   Feature 1: At 0.18 function slope changes from -ve to +ve, at 0.86 goes -ve again.
    -   Feature 2: At 0.20 function slope changes from -ve to +ve, at 0.86 goes -ve again.
-   Function 3: At 0.38 for all 3 features, function slope changes from -ve to +ve and at 0.825 output declines sharply.
-   Function 4: At 0.37 for all 4 features, function slope changes sharply from +ve to -ve.
-   Function 5: At 0.49 for all 4 features, output sharply rises from low values (< 10) and rapidly climbs above 8500.
-   Function 6: At 0.42 for all 6 features, function slope changes from +ve to -ve.
-   Function 7: At 0.30 for all 6 features, function slope changes from +ve to -ve, at 0.86 goes +ve again.
-   Function 8: At 0.215 for all 8 features, function slope changes from +ve to -ve.

#### If you trained a neural network or another surrogate model, did you explore how the outputs change in response to the inputs How might these gradients point to directions that reduce the function value If you did not train a neural network or surrogate model, explain why you chose not to.

-   I used **Gaussian Process** as the surrogate model, not **Neural Network**s. A Neural Network would require more data to train reliably and does not
    naturally output a well-calibrated uncertainty estimate. With roughly 15–30 observations per function, a GP is decidedly more data-efficient and produces
    the mean output prediction as well as the uncertainty prediction at every input data-point.

#### Imagine framing your BBO capstone project as a classification task (‘good’ vs ‘bad’ outputs). How could models such as logistic regression, SVMs or neural networks capture this decision boundary What trade-offs would you face between misclassification and exploration

-   Logistic regression computes the probability of an output variable to belong or not belong to a particular class. The numerical value of the probability
    determines the inclusion or exclusion decision - for example, high probability (near 1.0) and low probability (near 0.0) may classify the output as 'good'
    and 'bad' respectively.
-   A Support Vector Machine (SVM) performs the classification task by finding the separator line or plane or hyperplane that can segregate different classes of
    outputs most efficiently. If there are multiple candidate separator surfaces, SVM will select the one with maximum margin on both sides of the surface.

#### Which type of model – linear regression, SVM or neural network – felt most appropriate for guiding your search How did you balance interpretability against flexibility when making this choice

-   I used **Gaussian Process** as the surrogate model, but when comparing the three models mentioned: Linear Regression, SVM, and Neural Networks an SVM-like
    logic using RBF kernels can be a close approximation to the needs of Bayesian Optimization.

#### Looking at your neural network surrogate, which input variables showed the steepest gradients or the greatest influence on your predictions How might you use this to prioritise your next experiments

-   I used **Gaussian Process** as the surrogate model, not **Neural Network**s.
-   I have added a new capability: permutation importance, that can calculate the relative importance of individual features in terms of effect on the output.
    According to this technique, the most impactful or influential features per function are:
    -   Function 1: Has 2 input features, both are almost equally influential.
    -   Function 2: Has 2 input features, feature 2 is slightly more influential than feature 1.
    -   Function 3: Has 3 input features, feature 1 is most influential, followed by features 2 & 3 as distant 2nd & 3rd respectively.
    -   Function 4: Has 4 input features, feature 1 is most influential, followed by features 2, 3, 4 gradually decreasing in influence.
    -   Function 5: Has 4 input features, feature 3 is most influential, followed by features 1, 4, 2 gradually decreasing in influence.
    -   Function 6: Has 5 input features, feature 4 is most influential, followed by feature 5 as distant 2nd. The remaining features have very little
        influence.
    -   Function 7: Has 6 input features, feature 6 is most influential, followed by features 1, 4 & 5 as distant 2nd, 3rd & 4th respectively. The remaining
        features have very little influence.
    -   Function 8: Has 8 input features, feature 3 is most influential, followed by feature 1 as 2nd. Features 4, 7 & 2 rank as distant 3rd, 4th & 5th
        respectively. The remaining features have very little influence.
-   Feature importance is a useful knowledge that can guide us to select the next query point for a given function. When we have a fair idea which features are
    most impactful, we can choose query points where the function is likely to yield high values with respect to these influential features.

#### When framing your BBO problem as a classification task (‘good’ vs ‘bad’ outputs), how effectively did your neural network approximate the decision boundary In what ways did backpropagation help you interpret or visualise this boundary

-   I used **Gaussian Process** as the surrogate model, not **Neural Network**s. However, if Neural Network was used, the partial derivatives of the output with
    respect to each input could have been computed through backpropagation. This could have helped visualisation of the edges of the high-output regions.

#### Compared to simpler models such as linear or logistic regression, how well did your neural network capture non-linear patterns in the function Was the added flexibility worth the extra complexity in tuning and interpretation

-   I used **Gaussian Process** as the surrogate model, not **Neural Network**s.
