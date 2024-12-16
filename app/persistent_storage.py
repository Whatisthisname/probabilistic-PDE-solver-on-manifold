import shelve
import jax.numpy as jnp


def set_value(key, value, filename="mydata.db"):
    """Set a value in the persistent dictionary."""
    with shelve.open(filename, writeback=True) as db:
        db[key] = value


def get_value(key, default=None, filename="mydata.db"):
    """Get a value from the persistent dictionary."""
    with shelve.open(filename) as db:
        return db.get(key, default)


def remove_value(key, filename="mydata.db"):
    """Remove a value from the persistent dictionary."""
    with shelve.open(filename, writeback=True) as db:
        if key in db:
            del db[key]
        else:
            print(f"Key '{key}' not found.")


def wipe_db(filename="mydata.db"):
    """Wipe all data from the persistent dictionary."""
    with shelve.open(filename, writeback=True) as db:
        db.clear()


def build_experiment_name(prior, derivatives, timesteps):
    return f"{prior}_{derivatives}_{timesteps}"


def tan(x):
    return jnp.sin(x) / jnp.cos(x)


# choice = "heat and square"
choice = "wave and tan"

experiment_setup = dict()
experiment_setup["heat and square"] = {
    "problem_title": "∂u/∂t = -Δu -u²",
    "priors": ["heat", "iwp"],
    "prior_scale": [2],
    "derivatives": [1, 2, 3, 4],
    "timesteps": jnp.logspace(1, 3.5, 10, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, laplace: -laplace @ u - u**2,
    "order": 1,
}
experiment_setup["wave and tan"] = {
    "problem_title": "∂²u/∂t² = -Δu -tan(u)",
    "priors": ["wave", "iwp"],
    "prior_scale": [2.5],
    "derivatives": [1, 2, 3],
    "timesteps": jnp.logspace(1, 3.5, 10, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, du, laplace: -(laplace @ u) - 1e-10 * tan(u),
    "order": 2,
}
