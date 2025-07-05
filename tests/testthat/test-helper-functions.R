test_that("luz_model_sequential creates proper sequential models", {
  skip_if_not_installed("luz")
  skip_if_not_installed("torch")
  skip_on_cran()
  
  # Test basic sequential model creation
  model <- luz_model_sequential(
    torch::nn_linear(5, 10),
    torch::nn_relu(),
    torch::nn_linear(10, 1)
  )
  
  expect_s3_class(model, "nn_module_generator")
  
  # Instantiate the model
  model_instance <- model()
  expect_s3_class(model_instance, "nn_sequential")
  expect_s3_class(model_instance, "nn_module")
})

test_that("luz_model_sequential handles various layer configurations", {
  skip_if_not_installed("luz")
  skip_if_not_installed("torch")
  skip_on_cran()
  
  # Test with different activation functions
  model1 <- luz_model_sequential(
    torch::nn_linear(3, 5),
    torch::nn_tanh(),
    torch::nn_linear(5, 2),
    torch::nn_sigmoid(),
    torch::nn_linear(2, 1)
  )
  
  # Test with single layer
  model2 <- luz_model_sequential(
    torch::nn_linear(1, 1)
  )
  
  # Test with many layers
  model3 <- luz_model_sequential(
    torch::nn_linear(10, 20),
    torch::nn_relu(),
    torch::nn_linear(20, 15),
    torch::nn_leaky_relu(),
    torch::nn_linear(15, 10),
    torch::nn_softplus(),
    torch::nn_linear(10, 5),
    torch::nn_tanh(),
    torch::nn_linear(5, 1)
  )
  
  expect_s3_class(model1(), "nn_sequential")
  expect_s3_class(model2(), "nn_sequential")
  expect_s3_class(model3(), "nn_sequential")
})

test_that("luz_model_sequential_check validates model types correctly", {
  skip_if_not_installed("luz")
  skip_if_not_installed("torch")
  skip_on_cran()
  
  # Valid sequential model
  valid_model <- luz_model_sequential(
    torch::nn_linear(2, 3),
    torch::nn_relu(),
    torch::nn_linear(3, 1)
  )
  
  expect_no_error(luz_model_sequential_check(valid_model))
  expect_no_error(luz_model_sequential_check(valid_model()))
  
  # Invalid model type
  invalid_model <- torch::nn_linear(2, 1)
  expect_error(luz_model_sequential_check(invalid_model))
})

test_that("luz_model_sequential forward pass works correctly", {
  skip_if_not_installed("luz")
  skip_if_not_installed("torch")
  skip_on_cran()
  
  model <- luz_model_sequential(
    torch::nn_linear(3, 5),
    torch::nn_relu(),
    torch::nn_linear(5, 1)
  )()
  
  # Test forward pass with appropriate input
  input_tensor <- torch::torch_randn(10, 3)
  output <- model(input_tensor)
  
  expect_s3_class(output, "torch_tensor")
  expect_equal(output$shape, c(10, 1))
})

test_that("torch package requirement is properly handled", {
  # Mock missing torch package scenario
  with_mocked_bindings(
    requireNamespace = function(pkg, quietly = TRUE) {
      if (pkg == "torch") return(FALSE)
      return(TRUE)
    },
    {
      expect_error(
        luz_model_sequential(list()),
        "package 'torch' is required"
      )
    }
  )
})