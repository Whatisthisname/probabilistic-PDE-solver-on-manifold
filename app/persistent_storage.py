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


experiment_setup = dict()
experiment_setup["wave"] = {
    "problem_title": "$∂²u/∂t² = -Δu$",
    "priors": ["wave", "iwp"],
    "prior_scale": [2.5],
    "derivatives": [2, 3, 4],
    "timesteps": jnp.logspace(1, 3.5, 10, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, du, laplace: -(laplace @ u),
    "order": 2,
    "domain": "icosphere",
}
experiment_setup["wave and tiny tan"] = {
    "problem_title": "$∂²u/∂t² = -Δu -10^{-6}tan(u)$",
    "priors": ["wave", "iwp"],
    "prior_scale": [2.5],
    "derivatives": [2, 3, 4],
    "timesteps": jnp.logspace(1, 3.5, 10, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, du, laplace: -(laplace @ u) - 1e-6 * tan(u),
    "order": 2,
    "domain": "icosphere",
}
experiment_setup["wave and medium tan"] = {
    "problem_title": "$∂²u/∂t² = -Δu -10^{-3} tan(u)$",
    "priors": ["wave", "iwp"],
    "prior_scale": [2.5],
    "derivatives": [2, 3, 4],
    "timesteps": jnp.logspace(1, 3.5, 10, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, du, laplace: -(laplace @ u) - 1e-3 * tan(u),
    "order": 2,
    "domain": "icosphere",
}
experiment_setup["wave and big tan"] = {
    "problem_title": "$∂²u/∂t² = -Δu -tan(u)$",
    "priors": ["wave", "iwp"],
    "prior_scale": [2.5],
    "derivatives": [2, 3, 4],
    "timesteps": jnp.logspace(1, 3.5, 10, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, du, laplace: -(laplace @ u) - tan(u),
    "order": 2,
    "domain": "icosphere",
}
experiment_setup["heat and tiny square"] = {
    "problem_title": "$∂u/∂t = -Δu -10^{-6}u²$",
    "priors": ["heat", "iwp"],
    "prior_scale": [2],
    "derivatives": [1, 2, 3, 4],
    "timesteps": jnp.logspace(1, 3.5, 10, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, laplace: -laplace @ u - 1e-6 * u**2,
    "order": 1,
    "domain": "icosphere",
}
experiment_setup["heat and medium square"] = {
    "problem_title": "$∂u/∂t = -Δu -10^{-3}u²$",
    "priors": ["heat", "iwp"],
    "prior_scale": [2],
    "derivatives": [1, 2, 3, 4],
    "timesteps": jnp.logspace(1, 3.5, 10, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, laplace: -laplace @ u - 1e-3 * u**2,
    "order": 1,
    "domain": "icosphere",
}
experiment_setup["heat and big square"] = {
    "problem_title": "$∂u/∂t = -Δu -u²$",
    "priors": ["heat", "iwp"],
    "prior_scale": [2],
    "derivatives": [1, 2, 3, 4],
    "timesteps": jnp.logspace(1, 3.5, 10, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, laplace: -laplace @ u - u**2,
    "order": 1,
    "domain": "icosphere",
}
experiment_setup["heat"] = {
    "problem_title": "$∂u/∂t = -Δu$",
    "priors": ["heat", "iwp"],
    "prior_scale": [2],
    "derivatives": [1, 2, 3, 4],
    "timesteps": jnp.logspace(1, 3.5, 10, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, laplace: -laplace @ u,
    "order": 1,
    "domain": "icosphere",
}
experiment_setup["torus heat PDE"] = {
    "problem_title": "$∂u/∂t = -(Δu)$",
    "priors": ["iwp"],
    "prior_scale": [2],
    "derivatives": [1, 2],
    "timesteps": jnp.logspace(1, 3.5, 5, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, laplace: -(laplace @ u),  # * (1 - u**2),
    "order": 1,
    "domain": "cut_torus",  # .obj
}
experiment_setup["torus nonlinear heat PDE"] = {
    "problem_title": r"$∂u/∂t = -(Δu)\cdot(1-u²)$",
    "priors": ["iwp"],
    "prior_scale": [2],
    "derivatives": [1, 2],
    "timesteps": jnp.logspace(1, 3.5, 5, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, laplace: -(laplace @ u) * (1 - u**2),
    "order": 1,
    "domain": "cut_torus",  # .obj
}
experiment_setup["torus wave PDE"] = {
    "problem_title": "$∂²u/∂t² = -Δu$",
    "priors": ["iwp"],
    "prior_scale": [2],
    "derivatives": [2, 3],
    "timesteps": jnp.logspace(1, 3.5, 5, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, du, laplace: -(laplace @ u),
    "order": 2,
    "domain": "cut_torus",  # .obj
}
experiment_setup["torus nonlinear wave PDE"] = {
    "problem_title": "$∂²u/∂t² = -Δu - (∂u/∂t)^3$",
    "priors": ["iwp"],
    "prior_scale": [2],
    "derivatives": [2, 3],
    "timesteps": jnp.logspace(1, 3.5, 5, endpoint=True, base=10).astype(int),
    "vector_field": lambda u, du, laplace: -(laplace @ u) - du**3,
    "order": 2,
    "domain": "cut_torus",  # .obj
}
