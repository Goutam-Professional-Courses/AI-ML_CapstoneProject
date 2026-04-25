#### What was the main principle or heuristic you used to decide on each query point (e.g. exploitation of high outputs, exploration of uncertain regions, diversity of samples)?

* This is the 1st week of query submission. In early stages of Bayesian optimisation, exploration should be preferred over exploitation as most the possible outputs of the true, objective function are not known. In such cases, we can form our query based on data-points at random or some average/mid-way value. I have chosen the latter (that is, the mid-point within the 2-D, 3-D or higher-dimensional hyperspace) across all 8 objective function as I feel this has a better chance to detect the broad-level shape of these functions than querying random points.

#### Which function(s) were most challenging to query, and why? What additional information would have helped you?

* Function 1 was most challenging, because the output values extremely small, making it practically impossible to plot on a scatter chart.

#### How do you plan to adjust your strategy in future rounds based on the current performance or uncertainty levels?

* If the output for the queried data-points move towards higher (because this is a maximisation challenge) than current values, emphasis on exploration can be reduced to allocate more resources towards exploitation - that is, search more in the vicinity of the (newly discovered) updated data-point. However, if the output drops or stays more or less the same, we will continue to aggressively explore the rest of the function space.
* For noisy functions, the search radius can be gradually expanded to better map the space.

