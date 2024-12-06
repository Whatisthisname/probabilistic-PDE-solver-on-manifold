import shelve


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


experiment_setup = dict()
experiment_setup["heat and tanh"] = {
    "problem_title": "∂u/∂t = -Δu -tanh(Δu)",
    "dbname": "heat_and_tanh",
    "priors": ["heat", "iwp"],
    "derivatives": [1, 2, 3, 4],
    "timesteps": [
        30,
        50,
        75,
        100,
        200,
        300,
        400,
        600,
        800,
        1000,
        1200,
        1600,
        2000,
        2400,
        2800,
        3200,
        3600,
        4000,
    ][:10],
}
experiment_setup["heat small tanh"] = {
    "problem_title": "∂u/∂t = -Δu -0.1*tanh(Δu)",
    "dbname": "heat_small_tanh",
    "priors": ["heat", "iwp"],
    "derivatives": [1, 2],
    "timesteps": [
        30,
        50,
        75,
        100,
        200,
        300,
        400,
        600,
        800,
        1000,
        1200,
        1600,
        2000,
        2400,
        2800,
        3200,
        3600,
        4000,
    ],
}
experiment_setup["heat"] = {
    "problem_title": "∂u/∂t = -Δu",
    "dbname": "heat",
    "priors": ["heat", "iwp"],
    "derivatives": [1, 2],
    "timesteps": [
        30,
        50,
        75,
        100,
        200,
        300,
        400,
        600,
        800,
        1000,
        1200,
        1600,
        2000,
        2400,
        2800,
        3200,
        3600,
        4000,
    ],
}
experiment_setup["wave"] = {
    "problem_title": "∂²u/∂t² = -Δu",
    "dbname": "wave",
    "priors": ["wave", "iwp"],
    "derivatives": [2],
    "timesteps": [
        30,
        50,
        75,
        100,
        200,
        300,
        400,
        600,
        800,
        1000,
        1200,
        1600,
        2000,
        2400,
        2800,
        3200,
        3600,
        4000,
    ],
}
experiment_setup["wave and tanh"] = {
    "problem_title": "∂²u/∂t² = -Δu - tanh(Δu)",
    "dbname": "wave_and_tanh",
    "priors": ["wave", "iwp"],
    "derivatives": [2],
    "timesteps": [
        30,
        50,
        75,
        100,
        200,
        300,
        400,
        600,
        800,
        1000,
        1200,
        1600,
        2000,
        2400,
        2800,
        3200,
        3600,
        4000,
    ],
}
