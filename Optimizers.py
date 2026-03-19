# Works by updating the value of weights to reduce the loss

# base gradient descent works with:

# weight = weight - learning rate * gradient

# weight acts as the position you are

# learning rate acts as how big of a step you take

# gradient acts as the slope of the mountain

# together learning rate and gradients define how big of a step you take in a direction which influences the weight

# Stochastic gradient descent works by:
# -instead of using all data points (loss values of data points)
# -it uses batches of losses and runs gd on that
# -this improves training speed/efficiency

# Adam works by:
# -adding calculations to the gradient to include momentum this helps incentivize lowering the loss
# m = beta1 * m + (1 - beta1) * gradient         momentum — smoothed gradient direction
# v = beta2 * v + (1 - beta2) * gradient²        velocity — smoothed gradient magnitude
# weight = weight - learning_rate * m / sqrt(v)
# - m tracks gradient direction (momentum) — smooths out noisy zigzagging
# - v tracks gradient magnitude — gives each weight its own adaptive learning rate
# - active weights (large gradients) get smaller steps
# - inactive weights (small gradients) get larger steps