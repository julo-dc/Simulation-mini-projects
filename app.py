from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import simpy
import math
import random

app = Flask(__name__)
CORS(app)

# Model parameters that can be drawn from a user-chosen distribution.
SAMPLABLE_PARAMS = (
    'initial_sheep',
    'initial_wolves',
    'sheep_birth_rate',
    'conversion_efficiency',
    'sheep_lifespan',
    'wolf_lifespan',
    'predation_rate',
    'carrying_capacity',
    'refuge_size',
    'disease_factor',
    'environmental_stress',
    'sheep_competition',
    'wolf_competition',
    'migration_rate',
)

PARAM_DEFAULTS = {
    'initial_sheep': 100,
    'initial_wolves': 20,
    'sheep_birth_rate': 0.6,
    'conversion_efficiency': 0.2,
    'sheep_lifespan': 11.0,
    'wolf_lifespan': 13.0,
    'predation_rate': 0.1,
    'carrying_capacity': 800,
    'refuge_size': 10.0,
    'disease_factor': 0.0,
    'environmental_stress': 0.0,
    'sheep_competition': 0.0,
    'wolf_competition': 0.0,
    'migration_rate': 0.0,
}

# Soft bounds used when clamping sampled values.
PARAM_BOUNDS = {
    'initial_sheep': (1, 5000),
    'initial_wolves': (1, 2000),
    'sheep_birth_rate': (0.0, 5.0),
    'conversion_efficiency': (0.0, 1.0),
    'sheep_lifespan': (0.1, 100.0),
    'wolf_lifespan': (0.1, 100.0),
    'predation_rate': (0.0, 5.0),
    'carrying_capacity': (1, 20000),
    'refuge_size': (0.0, 5000.0),
    'disease_factor': (0.0, 1.0),
    'environmental_stress': (0.0, 1.0),
    'sheep_competition': (0.0, 1.0),
    'wolf_competition': (0.0, 1.0),
    'migration_rate': (-1.0, 1.0),
}

INTEGER_PARAMS = {'initial_sheep', 'initial_wolves', 'carrying_capacity'}


def _as_float(value, default=0.0):
    try:
        if value is None or value == '':
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value, low, high):
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def normalize_param_spec(name, raw):
    """Normalize a parameter into {dist, ...args}.

    Accepts either a bare number (fixed) or a distribution object.
    """
    default = PARAM_DEFAULTS[name]
    if raw is None:
        return {'dist': 'fixed', 'value': default}
    if isinstance(raw, (int, float)):
        return {'dist': 'fixed', 'value': float(raw)}
    if not isinstance(raw, dict):
        return {'dist': 'fixed', 'value': default}

    dist = str(raw.get('dist', 'fixed')).lower().strip()
    if dist in ('constant', 'deterministic', 'none', ''):
        dist = 'fixed'

    if dist == 'fixed':
        return {'dist': 'fixed', 'value': _as_float(raw.get('value', default), default)}
    if dist == 'uniform':
        low = _as_float(raw.get('min', raw.get('low', default)), default)
        high = _as_float(raw.get('max', raw.get('high', default)), default)
        if high < low:
            low, high = high, low
        return {'dist': 'uniform', 'min': low, 'max': high}
    if dist == 'normal':
        return {
            'dist': 'normal',
            'mean': _as_float(raw.get('mean', default), default),
            'std': max(0.0, _as_float(raw.get('std', 0.0), 0.0)),
            'min': raw.get('min', None),
            'max': raw.get('max', None),
        }
    if dist == 'lognormal':
        # mu/sigma are parameters of the underlying normal for ln(X).
        # If only mean/sigma are provided, convert mean to mu so E[X] ~= mean.
        sigma = max(1e-12, _as_float(raw.get('sigma', raw.get('std', 0.1)), 0.1))
        if 'mu' in raw and raw['mu'] is not None and raw['mu'] != '':
            mu = _as_float(raw.get('mu'), math.log(max(default, 1e-12)))
        else:
            mean = max(1e-12, _as_float(raw.get('mean', default), default))
            mu = math.log(mean) - 0.5 * sigma * sigma
        return {
            'dist': 'lognormal',
            'mu': mu,
            'sigma': sigma,
            'min': raw.get('min', None),
            'max': raw.get('max', None),
        }
    if dist == 'triangular':
        low = _as_float(raw.get('min', raw.get('low', default)), default)
        high = _as_float(raw.get('max', raw.get('high', default)), default)
        mode = _as_float(raw.get('mode', default), default)
        if high < low:
            low, high = high, low
        mode = _clamp(mode, low, high)
        return {'dist': 'triangular', 'min': low, 'mode': mode, 'max': high}

    return {'dist': 'fixed', 'value': default}


def sample_param(name, spec, rng):
    """Draw one value for a named parameter from its distribution spec."""
    spec = normalize_param_spec(name, spec)
    bound_low, bound_high = PARAM_BOUNDS[name]
    dist = spec['dist']

    if dist == 'fixed':
        value = spec['value']
    elif dist == 'uniform':
        value = rng.uniform(spec['min'], spec['max'])
    elif dist == 'normal':
        value = rng.gauss(spec['mean'], spec['std'])
        clip_min = spec['min'] if spec['min'] is not None and spec['min'] != '' else bound_low
        clip_max = spec['max'] if spec['max'] is not None and spec['max'] != '' else bound_high
        value = _clamp(value, _as_float(clip_min, bound_low), _as_float(clip_max, bound_high))
    elif dist == 'lognormal':
        value = rng.lognormvariate(spec['mu'], spec['sigma'])
        clip_min = spec['min'] if spec['min'] is not None and spec['min'] != '' else bound_low
        clip_max = spec['max'] if spec['max'] is not None and spec['max'] != '' else bound_high
        value = _clamp(value, _as_float(clip_min, bound_low), _as_float(clip_max, bound_high))
    elif dist == 'triangular':
        value = rng.triangular(spec['min'], spec['max'], spec['mode'])
    else:
        value = PARAM_DEFAULTS[name]

    value = _clamp(value, bound_low, bound_high)
    if name in INTEGER_PARAMS:
        return int(round(value))
    return float(value)


def is_random_spec(spec):
    if isinstance(spec, dict):
        return str(spec.get('dist', 'fixed')).lower() not in (
            'fixed', 'constant', 'deterministic', 'none', ''
        )
    return False


def sample_parameters(param_specs, rng):
    """Sample a full parameter set for one simulation run."""
    sampled = {}
    for name in SAMPLABLE_PARAMS:
        sampled[name] = sample_param(name, param_specs.get(name), rng)
    return sampled


class Ecosystem:
    def __init__(self, env, initial_sheep, initial_wolves,
                 sheep_birth_rate, conversion_efficiency,
                 sheep_lifespan, wolf_lifespan,
                 predation_rate, carrying_capacity,
                 refuge_size=0.0,
                 disease_factor=0.0, environmental_stress=0.0,
                 sheep_competition=0.0, wolf_competition=0.0,
                 migration_rate=0.0):
        self.env = env
        self.sheep_count = float(initial_sheep)
        self.wolf_count = float(initial_wolves)
        self.sheep_birth_rate = sheep_birth_rate
        self.conversion_efficiency = conversion_efficiency
        self.sheep_lifespan = sheep_lifespan
        self.wolf_lifespan = wolf_lifespan
        self.predation_rate = predation_rate
        self.carrying_capacity = carrying_capacity
        self.handling_time = 0.1
        self.refuge_size = refuge_size
        self.disease_factor = disease_factor
        self.environmental_stress = environmental_stress
        self.sheep_competition = sheep_competition
        self.wolf_competition = wolf_competition
        self.migration_rate = migration_rate
        self.history = []
        self.dt = 0.01

        self.sheep_avg_age = 0.0
        self.wolf_avg_age = 0.0

    def calculate_predation_rate(self, sheep, wolves):
        """Holling Type II functional response with prey refuge."""
        if sheep <= 0 or wolves <= 0:
            return 0.0

        effective_sheep = max(0.0, sheep - self.refuge_size)
        if effective_sheep <= 0:
            return 0.0

        denominator = 1.0 + self.handling_time * effective_sheep
        predation = (self.predation_rate * effective_sheep * wolves) / denominator
        return min(predation, effective_sheep)

    def calculate_age_based_mortality(self, age, lifespan):
        """Logistic mortality rate based on age relative to lifespan."""
        if lifespan <= 0:
            return 1.0

        k = 2.0
        normalized_age = age / lifespan if lifespan > 0 else 1.0
        mortality_rate = 1.0 / (1.0 + math.exp(-k * (normalized_age - 1.0)))
        return min(mortality_rate * 0.5, 0.5)

    def compute_derivatives(self, sheep, wolves):
        """Compute derivatives dSheep/dt and dWolf/dt for the current state."""
        sheep = max(0.0, sheep)
        wolves = max(0.0, wolves)

        logistic_growth = 0.0
        if sheep > 1 and self.carrying_capacity > 0:
            logistic_growth = self.sheep_birth_rate * sheep * (1.0 - sheep / self.carrying_capacity)
            logistic_growth = max(0.0, logistic_growth)

        sheep_mortality_rate = self.calculate_age_based_mortality(self.sheep_avg_age, self.sheep_lifespan)
        sheep_death = sheep * sheep_mortality_rate
        disease_death = sheep * self.disease_factor
        competition_death = self.sheep_competition * sheep * sheep
        stress_death = sheep * self.environmental_stress * 0.5
        predation_rate = self.calculate_predation_rate(sheep, wolves)
        migration = sheep * self.migration_rate

        dSheep = (
            logistic_growth - sheep_death - predation_rate
            - disease_death - competition_death - stress_death + migration
        )

        wolf_birth = 0.0
        if wolves > 1 and predation_rate > 0:
            wolf_birth = predation_rate * self.conversion_efficiency

        wolf_mortality_rate = self.calculate_age_based_mortality(self.wolf_avg_age, self.wolf_lifespan)
        wolf_death = wolves * wolf_mortality_rate

        if predation_rate == 0 and wolves > 0:
            wolf_death = wolves

        wolf_disease_death = wolves * self.disease_factor
        wolf_competition_death = self.wolf_competition * wolves * wolves
        wolf_stress_death = wolves * self.environmental_stress * 0.5
        wolf_migration = wolves * self.migration_rate

        dWolf = (
            wolf_birth - wolf_death - wolf_disease_death
            - wolf_competition_death - wolf_stress_death + wolf_migration
        )
        return (dSheep, dWolf)

    def runge_kutta_step(self, sheep, wolves):
        """Perform one Runge-Kutta 4 (RK4) integration step."""
        dt = self.dt

        k1_sheep, k1_wolf = self.compute_derivatives(sheep, wolves)
        k2_sheep, k2_wolf = self.compute_derivatives(
            sheep + 0.5 * dt * k1_sheep,
            wolves + 0.5 * dt * k1_wolf
        )
        k3_sheep, k3_wolf = self.compute_derivatives(
            sheep + 0.5 * dt * k2_sheep,
            wolves + 0.5 * dt * k2_wolf
        )
        k4_sheep, k4_wolf = self.compute_derivatives(
            sheep + dt * k3_sheep,
            wolves + dt * k3_wolf
        )

        dSheep = (dt / 6.0) * (k1_sheep + 2 * k2_sheep + 2 * k3_sheep + k4_sheep)
        dWolf = (dt / 6.0) * (k1_wolf + 2 * k2_wolf + 2 * k3_wolf + k4_wolf)

        return (max(0.0, sheep + dSheep), max(0.0, wolves + dWolf))

    def update_populations(self):
        """Update populations using RK4 integration, one year at a time."""
        steps_per_year = int(1.0 / self.dt)

        while True:
            for _ in range(steps_per_year):
                self.sheep_count, self.wolf_count = self.runge_kutta_step(
                    self.sheep_count,
                    self.wolf_count
                )

                if self.sheep_count > 0:
                    logistic_growth = 0.0
                    if self.sheep_count > 1 and self.carrying_capacity > 0:
                        logistic_growth = (
                            self.sheep_birth_rate * self.sheep_count
                            * (1.0 - self.sheep_count / self.carrying_capacity)
                        )
                        logistic_growth = max(0.0, logistic_growth)

                    birth_fraction = (logistic_growth * self.dt) / max(self.sheep_count, 1.0)
                    self.sheep_avg_age = (
                        (1.0 - birth_fraction) * (self.sheep_avg_age + self.dt)
                        + birth_fraction * 0.0
                    )
                    self.sheep_avg_age = max(0.0, self.sheep_avg_age)

                if self.wolf_count > 0:
                    predation_rate = self.calculate_predation_rate(self.sheep_count, self.wolf_count)
                    wolf_birth = 0.0
                    if self.wolf_count > 1 and predation_rate > 0:
                        wolf_birth = predation_rate * self.conversion_efficiency

                    birth_fraction = (wolf_birth * self.dt) / max(self.wolf_count, 1.0)
                    self.wolf_avg_age = (
                        (1.0 - birth_fraction) * (self.wolf_avg_age + self.dt)
                        + birth_fraction * 0.0
                    )
                    self.wolf_avg_age = max(0.0, self.wolf_avg_age)

                if self.sheep_count <= 1:
                    self.sheep_count = 0.0
                    self.sheep_avg_age = 0.0

                if self.wolf_count <= 1:
                    self.wolf_count = 0.0
                    self.wolf_avg_age = 0.0

                predation_rate_check = self.calculate_predation_rate(self.sheep_count, self.wolf_count)
                if predation_rate_check == 0 and self.wolf_count > 0:
                    self.wolf_count = 0.0
                    self.wolf_avg_age = 0.0

                if self.refuge_size > 0 and self.sheep_count > 1:
                    self.sheep_count = max(self.refuge_size, self.sheep_count)

            yield self.env.timeout(1)

    def record_state(self, record_interval=1):
        """Record current state of the ecosystem at specified intervals."""
        while True:
            yield self.env.timeout(record_interval)
            self.history.append({
                'time': self.env.now,
                'sheep': int(round(self.sheep_count)),
                'wolves': int(round(self.wolf_count))
            })


def run_simulation(initial_sheep=100, initial_wolves=20,
                  sheep_birth_rate=0.6, conversion_efficiency=0.2,
                  sheep_lifespan=11.0, wolf_lifespan=13.0,
                  predation_rate=0.1, carrying_capacity=800,
                  refuge_size=10.0,
                  duration=500, disease_factor=0.0,
                  environmental_stress=0.0, sheep_competition=0.0,
                  wolf_competition=0.0, migration_rate=0.0):
    """Run one deterministic simulation with the given parameter values."""
    env = simpy.Environment()
    ecosystem = Ecosystem(
        env, initial_sheep, initial_wolves,
        sheep_birth_rate, conversion_efficiency,
        sheep_lifespan, wolf_lifespan,
        predation_rate, carrying_capacity,
        refuge_size,
        disease_factor, environmental_stress,
        sheep_competition, wolf_competition,
        migration_rate
    )

    if duration <= 50:
        record_interval = 1
    elif duration <= 200:
        record_interval = 5
    else:
        record_interval = 10

    start_year = 2026 - duration

    ecosystem.history.append({
        'time': 0.0,
        'sheep': int(round(initial_sheep)),
        'wolves': int(round(initial_wolves))
    })

    env.process(ecosystem.update_populations())
    env.process(ecosystem.record_state(record_interval))
    env.run(until=duration)

    for entry in ecosystem.history:
        entry['time'] = int(round(start_year + entry['time']))

    return ecosystem.history


def extract_param_specs(data):
    """Build distribution specs from API payload (nested or flat)."""
    nested = data.get('parameters') if isinstance(data.get('parameters'), dict) else {}
    specs = {}
    for name in SAMPLABLE_PARAMS:
        if name in nested:
            specs[name] = nested[name]
        elif name in data:
            specs[name] = data[name]
        else:
            specs[name] = {'dist': 'fixed', 'value': PARAM_DEFAULTS[name]}
        specs[name] = normalize_param_spec(name, specs[name])
    return specs


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json or {}

    duration = int(data.get('duration', 500))
    num_runs = max(1, int(data.get('num_runs', 1)))
    seed = data.get('seed', None)
    if seed is not None and seed != '':
        seed = int(seed)
    else:
        seed = None

    param_specs = extract_param_specs(data)
    has_random = any(is_random_spec(spec) for spec in param_specs.values())
    if not has_random:
        num_runs = 1

    master_rng = random.Random(seed)
    all_runs = []
    sampled_params = []

    for _ in range(num_runs):
        run_seed = master_rng.randrange(1 << 30)
        run_rng = random.Random(run_seed)
        params = sample_parameters(param_specs, run_rng)
        sampled_params.append(params)
        results = run_simulation(duration=duration, **params)
        all_runs.append(results)

    if num_runs == 1:
        series = all_runs[0]
    else:
        series = []
        reference_times = [entry['time'] for entry in all_runs[0]]
        for i, ref_time in enumerate(reference_times):
            sheep_vals = [run[i]['sheep'] for run in all_runs]
            wolf_vals = [run[i]['wolves'] for run in all_runs]
            avg_sheep = sum(sheep_vals) / num_runs
            avg_wolves = sum(wolf_vals) / num_runs
            sheep_std = math.sqrt(sum((v - avg_sheep) ** 2 for v in sheep_vals) / num_runs)
            wolf_std = math.sqrt(sum((v - avg_wolves) ** 2 for v in wolf_vals) / num_runs)
            series.append({
                'time': int(ref_time),
                'sheep': int(round(avg_sheep)),
                'wolves': int(round(avg_wolves)),
                'sheep_std': round(sheep_std, 2),
                'wolves_std': round(wolf_std, 2),
            })

    return jsonify({
        'series': series,
        'probabilistic': has_random,
        'num_runs': num_runs,
        'parameter_specs': param_specs,
        'sampled_parameters': sampled_params if has_random else sampled_params[:1],
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
