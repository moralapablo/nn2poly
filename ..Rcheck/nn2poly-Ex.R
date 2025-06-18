pkgname <- "nn2poly"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('nn2poly')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("add_constraints")
### * add_constraints

flush(stderr()); flush(stdout())

### Name: add_constraints
### Title: Add constraints to a neural network
### Aliases: add_constraints

### ** Examples

## Not run:
##D if (requireNamespace("keras", quietly=TRUE)) {
##D   # ---- Example with a keras/tensorflow network ----
##D   # Build a small nn:
##D   nn <- keras::keras_model_sequential()
##D   nn <- keras::layer_dense(nn, units = 10, activation = "tanh", input_shape = 2)
##D   nn <- keras::layer_dense(nn, units = 1, activation = "linear")
##D
##D   # Add constraints
##D   nn_constrained <- add_constraints(nn, constraint_type = "l1_norm")
##D
##D   # Check that class of the constrained nn is "nn2poly"
##D   class(nn_constrained)[1]
##D }
##D
##D if (requireNamespace("luz", quietly=TRUE)) {
##D   # ---- Example with a luz/torch network ----
##D
##D   # Build a small nn
##D   nn <- luz_model_sequential(
##D     torch::nn_linear(2,10),
##D     torch::nn_tanh(),
##D     torch::nn_linear(10,1)
##D   )
##D
##D   # With luz/torch we need to setup the nn before adding the constraints
##D   nn <- luz::setup(module = nn,
##D     loss = torch::nn_mse_loss(),
##D     optimizer = torch::optim_adam,
##D   )
##D
##D   # Add constraints
##D   nn <- add_constraints(nn)
##D
##D   # Check that class of the constrained nn is "nn2poly"
##D   class(nn)[1]
##D }
## End(Not run)




cleanEx()
nameEx("luz_model_sequential")
### * luz_model_sequential

flush(stderr()); flush(stdout())

### Name: luz_model_sequential
### Title: Build a 'luz' model composed of a linear stack of layers
### Aliases: luz_model_sequential

### ** Examples

## Not run:
##D if (requireNamespace("luz", quietly=TRUE)) {
##D # Create a NN using luz/torch as a sequential model
##D # with 3 fully connected linear layers,
##D # the first one with input = 5 variables,
##D # 100 neurons and tanh activation function, the second
##D # one with 50 neurons and softplus activation function
##D # and the last one with 1 linear output.
##D nn <- luz_model_sequential(
##D   torch::nn_linear(5,100),
##D   torch::nn_tanh(),
##D   torch::nn_linear(100,50),
##D   torch::nn_softplus(),
##D   torch::nn_linear(50,1)
##D )
##D
##D nn
##D
##D # Check that the nn is of class nn_squential
##D class(nn)
##D }
## End(Not run)




cleanEx()
nameEx("nn2poly")
### * nn2poly

flush(stderr()); flush(stdout())

### Name: nn2poly
### Title: Obtain polynomial representation
### Aliases: nn2poly

### ** Examples

# Build a NN estructure with random weights, with 2 (+ bias) inputs,
# 4 (+bias) neurons in the first hidden layer with "tanh" activation
# function, 4 (+bias) neurons in the second hidden layer with "softplus",
# and 1 "linear" output unit

weights_layer_1 <- matrix(rnorm(12), nrow = 3, ncol = 4)
weights_layer_2 <- matrix(rnorm(20), nrow = 5, ncol = 4)
weights_layer_3 <- matrix(rnorm(5), nrow = 5, ncol = 1)

# Set it as a list with activation functions as names
nn_object = list("tanh" = weights_layer_1,
                 "softplus" = weights_layer_2,
                 "linear" = weights_layer_3)

# Obtain the polynomial representation (order = 3) of that neural network
final_poly <- nn2poly(nn_object, max_order = 3)

# Change the last layer to have 3 outputs (as in a multiclass classification)
# problem
weights_layer_4 <- matrix(rnorm(20), nrow = 5, ncol = 4)

# Set it as a list with activation functions as names
nn_object = list("tanh" = weights_layer_1,
                 "softplus" = weights_layer_2,
                 "linear" = weights_layer_4)
# Obtain the polynomial representation of that neural network
# In this case the output is formed by several polynomials with the same
# structure but different coefficient values
final_poly <- nn2poly(nn_object, max_order = 3)

# Polynomial representation of each hidden neuron is given by
final_poly <- nn2poly(nn_object, max_order = 3, keep_layers = TRUE)




cleanEx()
nameEx("plot.nn2poly")
### * plot.nn2poly

flush(stderr()); flush(stdout())

### Name: plot.nn2poly
### Title: Plot method for 'nn2poly' objects.
### Aliases: plot.nn2poly

### ** Examples

# --- Single polynomial output ---
# Build a NN structure with random weights, with 2 (+ bias) inputs,
# 4 (+bias) neurons in the first hidden layer with "tanh" activation
# function, 4 (+bias) neurons in the second hidden layer with "softplus",
# and 2 "linear" output units

weights_layer_1 <- matrix(rnorm(12), nrow = 3, ncol = 4)
weights_layer_2 <- matrix(rnorm(20), nrow = 5, ncol = 4)
weights_layer_3 <- matrix(rnorm(5), nrow = 5, ncol = 1)

# Set it as a list with activation functions as names
nn_object = list("tanh" = weights_layer_1,
                 "softplus" = weights_layer_2,
                 "linear" = weights_layer_3)

# Obtain the polynomial representation (order = 3) of that neural network
final_poly <- nn2poly(nn_object, max_order = 3)

# Plot all the coefficients, one plot per output unit
plot(final_poly)

# Plot only the 5 most important coeffcients (by absolute magnitude)
# one plot per output unit
plot(final_poly, n = 5)

# --- Multiple output polynomials ---
# Build a NN structure with random weights, with 2 (+ bias) inputs,
# 4 (+bias) neurons in the first hidden layer with "tanh" activation
# function, 4 (+bias) neurons in the second hidden layer with "softplus",
# and 2 "linear" output units

weights_layer_1 <- matrix(rnorm(12), nrow = 3, ncol = 4)
weights_layer_2 <- matrix(rnorm(20), nrow = 5, ncol = 4)
weights_layer_3 <- matrix(rnorm(10), nrow = 5, ncol = 2)

# Set it as a list with activation functions as names
nn_object = list("tanh" = weights_layer_1,
                 "softplus" = weights_layer_2,
                 "linear" = weights_layer_3)

# Obtain the polynomial representation (order = 3) of that neural network
final_poly <- nn2poly(nn_object, max_order = 3)

# Plot all the coefficients, one plot per output unit
plot(final_poly)

# Plot only the 5 most important coeffcients (by absolute magnitude)
# one plot per output unit
plot(final_poly, n = 5)




cleanEx()
nameEx("predict.nn2poly")
### * predict.nn2poly

flush(stderr()); flush(stdout())

### Name: predict.nn2poly
### Title: Predict method for 'nn2poly' objects.
### Aliases: predict.nn2poly

### ** Examples

# Build a NN structure with random weights, with 2 (+ bias) inputs,
# 4 (+bias) neurons in the first hidden layer with "tanh" activation
# function, 4 (+bias) neurons in the second hidden layer with "softplus",
# and 1 "linear" output unit

weights_layer_1 <- matrix(rnorm(12), nrow = 3, ncol = 4)
weights_layer_2 <- matrix(rnorm(20), nrow = 5, ncol = 4)
weights_layer_3 <- matrix(rnorm(5), nrow = 5, ncol = 1)

# Set it as a list with activation functions as names
nn_object = list("tanh" = weights_layer_1,
                 "softplus" = weights_layer_2,
                 "linear" = weights_layer_3)

# Obtain the polynomial representation (order = 3) of that neural network
final_poly <- nn2poly(nn_object, max_order = 3)

# Define some new data, it can be vector, matrix or dataframe
newdata <- matrix(rnorm(10), ncol = 2, nrow = 5)

# Predict using the obtained polynomial
predict(object = final_poly, newdata = newdata)

# Predict the values of each monomial of the obtained polynomial
predict(object = final_poly, newdata = newdata, monomials = TRUE)

# Change the last layer to have 3 outputs (as in a multiclass classification)
# problem
weights_layer_4 <- matrix(rnorm(20), nrow = 5, ncol = 4)

# Set it as a list with activation functions as names
nn_object = list("tanh" = weights_layer_1,
                 "softplus" = weights_layer_2,
                 "linear" = weights_layer_4)

# Obtain the polynomial representation of that neural network
# Polynomial representation of each hidden neuron is given by
final_poly <- nn2poly(nn_object, max_order = 3, keep_layers = TRUE)

# Define some new data, it can be vector, matrix or dataframe
newdata <- matrix(rnorm(10), ncol = 2, nrow = 5)

# Predict using the obtained polynomials (for all layers)
predict(object = final_poly, newdata = newdata)

# Predict using the obtained polynomials (for chosen layers)
predict(object = final_poly, newdata = newdata, layers = c(2,3))




### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
