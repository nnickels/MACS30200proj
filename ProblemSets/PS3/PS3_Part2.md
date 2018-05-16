PS3\_Part2
================
Nora Nickels
5/15/2018

Problem Set 3
=============

Part 2: Scalar Regression
-------------------------

### Monitor validation set performance on models while adjusting parameters

-   Use the Boston housing dataset from chapter 3.6 to predict median housing markets using a deep learning model
-   Use 10-fold cross validation to monitor validation set performance At the end of your notebook, report the test set MSE based on your final model trained using all of the training data

``` r
set.seed(1234)

library(keras)
# install_keras(tensorflow = "gpu")

dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

# Take a look at the data

str(train_data)
```

    ##  num [1:404, 1:13] 1.2325 0.0218 4.8982 0.0396 3.6931 ...

``` r
str(test_data)
```

    ##  num [1:102, 1:13] 18.0846 0.1233 0.055 1.2735 0.0715 ...

``` r
str(train_targets)
```

    ##  num [1:404(1d)] 15.2 42.3 50 21.1 17.7 18.5 11.3 15.6 15.6 14.4 ...

``` r
# Prep the data

mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)
```

### Try out a first model

``` r
# Because we will need to instantiate the same model multiple times,
# we use a function to construct it.

build_model_1 <- function() {
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", 
                input_shape = dim(train_data)[[2]]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1) 
    
  model %>% compile(
    optimizer = "rmsprop", 
    loss = "mse", 
    metrics = c("mae")
  )
}

k <- 10
indices <- sample(1:nrow(train_data))
folds <- cut(1:length(indices), breaks = k, labels = FALSE) 

num_epochs <- 100
all_scores <- c()
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE) 
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  # Build the Keras model (already compiled)
  model <- build_model_1()
  
  # Train the model (in silent mode, verbose=0)
  model %>% fit(partial_train_data, partial_train_targets,
                epochs = num_epochs, batch_size = 1, verbose = 0)
                
  # Evaluate the model on the validation data
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores <- c(all_scores, results$mean_absolute_error)
}  
```

    ## processing fold # 1 
    ## processing fold # 2 
    ## processing fold # 3 
    ## processing fold # 4 
    ## processing fold # 5 
    ## processing fold # 6 
    ## processing fold # 7 
    ## processing fold # 8 
    ## processing fold # 9 
    ## processing fold # 10

``` r
all_scores
```

    ##  [1] 2.086364 1.709044 2.893229 3.787378 1.842145 2.561610 2.827196
    ##  [8] 3.608388 2.354422 1.745185

``` r
mean(all_scores)
```

    ## [1] 2.541496

### Try out a second model, adjusting epochs

``` r
# Change number of epochs

num_epochs <- 70
all_mse_histories <- NULL

for (i in 1:k) {
  cat("processing fold #", i, "\n")
  
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  # Build the Keras model (already compiled)
  model <- build_model_1()
  
  # Train the model (in silent mode, verbose=0)
  history <- model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 1, verbose = 0
  )
  mse_history <- history$metrics$loss
  all_mse_histories <- rbind(all_mse_histories, mse_history)
}
```

    ## processing fold # 1 
    ## processing fold # 2 
    ## processing fold # 3 
    ## processing fold # 4 
    ## processing fold # 5 
    ## processing fold # 6 
    ## processing fold # 7 
    ## processing fold # 8 
    ## processing fold # 9 
    ## processing fold # 10

``` r
average_mse_history <- data.frame(
  epoch = seq(1:ncol(all_mse_histories)),
  validation_mse = apply(all_mse_histories, 2, mean)
)

library(ggplot2)

ggplot(average_mse_history, aes(x = epoch, y = validation_mse)) + geom_line()
```

![](PS3_Part2_files/figure-markdown_github/model%202-1.png)

``` r
ggplot(average_mse_history, aes(x = epoch, y = validation_mse)) + geom_smooth()
```

    ## `geom_smooth()` using method = 'loess'

![](PS3_Part2_files/figure-markdown_github/model%202-2.png)

### Try out a third model, adjusting neural net layer numbers

``` r
# Change model to have more layers

build_model_2 <- function() {
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", 
                input_shape = dim(train_data)[[2]]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1) 
    
  model %>% compile(
    optimizer = "rmsprop", 
    loss = "mse", 
    metrics = c("mae")
  )
}

# Some memory clean-up
k_clear_session()
num_epochs <- 80
all_mse_histories <- NULL

for (i in 1:k) {
  cat("processing fold #", i, "\n")
  
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  # Build the Keras model (already compiled)
  model <- build_model_2()
  
  # Train the model (in silent mode, verbose=0)
  history <- model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 1, verbose = 0
  )
  mse_history <- history$metrics$loss
  all_mse_histories <- rbind(all_mse_histories, mse_history)
}
```

    ## processing fold # 1 
    ## processing fold # 2 
    ## processing fold # 3 
    ## processing fold # 4 
    ## processing fold # 5 
    ## processing fold # 6 
    ## processing fold # 7 
    ## processing fold # 8 
    ## processing fold # 9 
    ## processing fold # 10

``` r
average_mse_history <- data.frame(
  epoch = seq(1:ncol(all_mse_histories)),
  validation_mse = apply(all_mse_histories, 2, mean)
)

library(ggplot2)

ggplot(average_mse_history, aes(x = epoch, y = validation_mse)) + geom_line()
```

![](PS3_Part2_files/figure-markdown_github/model%203-1.png)

``` r
ggplot(average_mse_history, aes(x = epoch, y = validation_mse)) + geom_smooth()
```

    ## `geom_smooth()` using method = 'loess'

![](PS3_Part2_files/figure-markdown_github/model%203-2.png)

### Report test set MSE based on final model trained using all the training data

``` r
# Get a fresh, compiled model.
model_final <- build_model_2()

# Train it on the entirety of the data.
model_final %>% fit(train_data, train_targets,
          epochs = 80, batch_size = 16, verbose = 0)

result_final <- model_final %>% 
                evaluate(test_data, test_targets)

result_final
```

    ## $loss
    ## [1] 16.90958
    ## 
    ## $mean_absolute_error
    ## [1] 2.873632
