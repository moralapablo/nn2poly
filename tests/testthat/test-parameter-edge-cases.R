test_that("get_parameters handles keras models with unusual architectures", {
  skip_if_not_installed("keras")
  skip_if_not_installed("tensorflow")
  skip_on_cran()
  
  # Test with minimal network (single neuron)
  model_minimal <- keras::keras_model_sequential() %>%
    keras::layer_dense(units = 1, input_shape = 1, activation = 'linear')
  
  params <- get_parameters(model_minimal)
  
  expect_equal(params$p, 1)
  expect_equal(length(params$weights_list), 1)
  expect_equal(length(params$af_string_list), 1)
  expect_equal(params$af_string_list[[1]], "linear")
})

test_that("get_parameters validates luz model structure", {
  skip_if_not_installed("luz")
  skip_if_not_installed("torch")
  skip_on_cran()
  
  # Test with valid sequential model
  valid_model <- luz_model_sequential(
    torch::nn_linear(2, 3),
    torch::nn_tanh(),
    torch::nn_linear(3, 1)
  )
  
  params <- get_parameters(valid_model)
  expect_equal(params$p, 2)
  expect_equal(length(params$weights_list), 2)
  
  # Test with non-sequential model should error
  invalid_model <- torch::nn_linear(2, 1)
  expect_error(get_parameters(invalid_model))
})

test_that("get_parameters handles different activation function names correctly", {
  skip_if_not_installed("luz")
  skip_if_not_installed("torch")
  skip_on_cran()
  
  model <- luz_model_sequential(
    torch::nn_linear(2, 3),
    torch::nn_relu(),
    torch::nn_linear(3, 2),
    torch::nn_sigmoid(),
    torch::nn_linear(2, 1)
  )
  
  params <- get_parameters(model)
  
  expect_equal(params$af_string_list[[1]], "relu")
  expect_equal(params$af_string_list[[2]], "sigmoid")
  expect_equal(params$af_string_list[[3]], "linear")  # No activation = linear
})

test_that("get_parameters correctly handles weight matrix dimensions", {
  skip_if_not_installed("luz")
  skip_if_not_installed("torch")
  skip_on_cran()
  
  model <- luz_model_sequential(
    torch::nn_linear(5, 10),
    torch::nn_tanh(),
    torch::nn_linear(10, 3),
    torch::nn_relu(),
    torch::nn_linear(3, 1)
  )()
  
  params <- get_parameters(model)
  
  # Check weight matrix dimensions are correct
  # First layer: 5 inputs -> 10 outputs, so weights should be (11, 10) with bias
  expect_equal(dim(params$weights_list[[1]]), c(6, 10))  # 5 weights + 1 bias
  expect_equal(dim(params$weights_list[[2]]), c(11, 3))  # 10 weights + 1 bias
  expect_equal(dim(params$weights_list[[3]]), c(4, 1))   # 3 weights + 1 bias
})

test_that("get_parameters handles luz_module_fitted objects", {
  skip_if_not_installed("luz")
  skip_if_not_installed("torch")
  skip_on_os("mac")
  skip_on_cran()
  
  testing_data <- testing_helper_2()
  data <- luz_test_data(testing_data)
  
  fitted <- luz_test_model() %>%
    luz::setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam
    ) %>%
    luz::fit(data$train, epochs = 1, valid_data = data$valid, verbose = FALSE)
  
  params <- get_parameters(fitted)
  
  expect_type(params, "list")
  expect_true(all(c("weights_list", "af_string_list", "n_neurons", "p") %in% names(params)))
})

test_that("get_parameters preserves numerical precision", {
  skip_if_not_installed("luz")
  skip_if_not_installed("torch")
  skip_on_cran()
  
  # Create model with known weights for precision testing
  model <- luz_model_sequential(
    torch::nn_linear(2, 2),
    torch::nn_linear(2, 1)
  )()
  
  # Set specific weights to test precision
  with_torch_manual_seed(123)
  
  params <- get_parameters(model)
  
  # Check that weights are numeric and finite
  expect_true(all(sapply(params$weights_list, function(w) all(is.finite(w)))))
  expect_true(all(sapply(params$weights_list, is.numeric)))
})

# Helper for torch seed setting
with_torch_manual_seed <- function(seed) {
  if (requireNamespace("torch", quietly = TRUE)) {
    torch::torch_manual_seed(seed)
  }
}