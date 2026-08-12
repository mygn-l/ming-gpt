import jax


def dropout(key, inputs, drop_rate):
    random_matrix = jax.random.uniform(key, inputs.shape) > drop_rate
    return (inputs * random_matrix) / (1 - drop_rate)