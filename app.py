from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import simpy
import math

app = Flask(__name__)
CORS(app)

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
        self.conversion_efficiency = conversion_efficiency  # How efficiently wolves convert eaten sheep into offspring
        self.sheep_lifespan = sheep_lifespan  # Expected lifespan in years
        self.wolf_lifespan = wolf_lifespan  # Expected lifespan in years
        self.predation_rate = predation_rate  # Combined attack rate and handling time for Holling Type II
        self.carrying_capacity = carrying_capacity
        self.handling_time = 0.1  # Fixed handling time for Holling Type II saturation
        self.refuge_size = refuge_size  # Prey refuge - sheep below this count are safe from predation
        self.disease_factor = disease_factor  # Additional mortality from disease
        self.environmental_stress = environmental_stress  # Stress affecting both populations
        self.sheep_competition = sheep_competition  # Intraspecific competition for sheep
        self.wolf_competition = wolf_competition  # Intraspecific competition for wolves
        self.migration_rate = migration_rate  # Net migration rate
        self.history = []
        self.dt = 0.01  # Time step for numerical integration (100 steps per year)
        
        # Track average age of populations (in years)
        self.sheep_avg_age = 0.0  # Start with newborns
        self.wolf_avg_age = 0.0  # Start with newborns
        
    def calculate_predation_rate(self, sheep, wolves):
        """Calculate predation rate using Holling Type II functional response with prey refuge
        
        Effective_Sheep = max(0, Sheep - Refuge_Size)
        Predation_Rate = (Predation_Rate * Effective_Sheep * Wolf) / (1 + Handling_Time * Effective_Sheep)
        """
        if sheep <= 0 or wolves <= 0:
            return 0.0
        
        # Prey refuge: only sheep above refuge_size are vulnerable to predation
        effective_sheep = max(0.0, sheep - self.refuge_size)
        
        if effective_sheep <= 0:
            return 0.0
        
        # Holling Type II: saturation occurs as effective sheep density increases
        # Using combined predation_rate parameter with fixed handling_time
        denominator = 1.0 + self.handling_time * effective_sheep
        predation = (self.predation_rate * effective_sheep * wolves) / denominator
        
        # Ensure predation doesn't exceed available effective sheep
        return min(predation, effective_sheep)
    
    def calculate_age_based_mortality(self, age, lifespan):
        """Calculate mortality rate using logistic function based on age
        
        Uses logistic function: mortality = 1 / (1 + exp(-k * (age - lifespan)))
        where k controls the steepness of the mortality curve
        """
        if lifespan <= 0:
            return 1.0  # If no lifespan, everything dies
        
        # Steepness parameter (higher = sharper transition at lifespan)
        k = 2.0
        
        # Logistic function: low mortality when young, increases as age approaches lifespan
        # Normalize age relative to lifespan
        normalized_age = age / lifespan if lifespan > 0 else 1.0
        
        # Logistic function: gives ~0 when age << lifespan, ~1 when age >> lifespan
        mortality_rate = 1.0 / (1.0 + math.exp(-k * (normalized_age - 1.0)))
        
        # Scale to reasonable annual mortality rate (max 0.5 per year to prevent instant extinction)
        return min(mortality_rate * 0.5, 0.5)
    
    def compute_derivatives(self, sheep, wolves):
        """Compute derivatives dSheep/dt and dWolf/dt for the current state
        
        Returns: (dSheep, dWolf)
        """
        sheep = max(0.0, sheep)
        wolves = max(0.0, wolves)
        
        # === 1. PREY (SHEEP) EQUATION ===
        # dSheep = (Sheep_Birth_Rate * Sheep * (1 - Sheep/Carrying_Capacity)) - (Sheep * Age_Based_Mortality) - Predation_Rate - Disease - Competition - Stress + Migration
        logistic_growth = 0.0
        # If population <= 1, cannot reproduce and will go extinct
        if sheep > 1 and self.carrying_capacity > 0:
            logistic_growth = self.sheep_birth_rate * sheep * (1.0 - sheep / self.carrying_capacity)
            logistic_growth = max(0.0, logistic_growth)  # Ensure non-negative
        
        # Age-based mortality using logistic function
        sheep_mortality_rate = self.calculate_age_based_mortality(self.sheep_avg_age, self.sheep_lifespan)
        sheep_death = sheep * sheep_mortality_rate
        
        # Disease mortality
        disease_death = sheep * self.disease_factor
        
        # Intraspecific competition (density-dependent mortality)
        competition_death = self.sheep_competition * sheep * sheep
        
        # Environmental stress
        stress_death = sheep * self.environmental_stress * 0.5
        
        # Calculate predation rate using Holling Type II with refuge
        predation_rate = self.calculate_predation_rate(sheep, wolves)
        
        # Migration (net immigration/emigration)
        migration = sheep * self.migration_rate
        
        # Total change in sheep population
        dSheep = logistic_growth - sheep_death - predation_rate - disease_death - competition_death - stress_death + migration
        
        # === 2. PREDATOR (WOLF) EQUATION ===
        # dWolf = (Predation_Rate * Conversion_Efficiency) - (Wolf * Age_Based_Mortality) - Disease - Competition - Stress + Migration
        # If population <= 1, cannot reproduce
        # If wolves are not eating sheep (predation_rate == 0), they die off
        wolf_birth = 0.0
        if wolves > 1 and predation_rate > 0:
            wolf_birth = predation_rate * self.conversion_efficiency
        
        wolf_mortality_rate = self.calculate_age_based_mortality(self.wolf_avg_age, self.wolf_lifespan)
        wolf_death = wolves * wolf_mortality_rate
        
        # If wolves are not eating sheep, they die off (starvation)
        if predation_rate == 0 and wolves > 0:
            wolf_death = wolves  # All wolves die from starvation
        
        # Disease mortality
        wolf_disease_death = wolves * self.disease_factor
        
        # Intraspecific competition (territory competition)
        wolf_competition_death = self.wolf_competition * wolves * wolves
        
        # Environmental stress
        wolf_stress_death = wolves * self.environmental_stress * 0.5
        
        # Migration (net immigration/emigration)
        wolf_migration = wolves * self.migration_rate
        
        dWolf = wolf_birth - wolf_death - wolf_disease_death - wolf_competition_death - wolf_stress_death + wolf_migration
        
        return (dSheep, dWolf)
    
    def runge_kutta_step(self, sheep, wolves):
        """Perform one Runge-Kutta 4 (RK4) integration step
        
        Returns: (new_sheep, new_wolves)
        """
        dt = self.dt
        
        # k1: derivative at current point
        k1_sheep, k1_wolf = self.compute_derivatives(sheep, wolves)
        
        # k2: derivative at midpoint using k1
        k2_sheep, k2_wolf = self.compute_derivatives(
            sheep + 0.5 * dt * k1_sheep,
            wolves + 0.5 * dt * k1_wolf
        )
        
        # k3: derivative at midpoint using k2
        k3_sheep, k3_wolf = self.compute_derivatives(
            sheep + 0.5 * dt * k2_sheep,
            wolves + 0.5 * dt * k2_wolf
        )
        
        # k4: derivative at endpoint using k3
        k4_sheep, k4_wolf = self.compute_derivatives(
            sheep + dt * k3_sheep,
            wolves + dt * k3_wolf
        )
        
        # Weighted average of derivatives
        dSheep = (dt / 6.0) * (k1_sheep + 2*k2_sheep + 2*k3_sheep + k4_sheep)
        dWolf = (dt / 6.0) * (k1_wolf + 2*k2_wolf + 2*k3_wolf + k4_wolf)
        
        # Update populations
        new_sheep = max(0.0, sheep + dSheep)
        new_wolves = max(0.0, wolves + dWolf)
        
        return (new_sheep, new_wolves)
    
    def update_populations(self):
        """Update populations using Runge-Kutta 4 (RK4) integration method"""
        steps_per_year = int(1.0 / self.dt)  # 100 steps per year (dt=0.01)
        
        while True:
            # Run integration steps for one year
            for _ in range(steps_per_year):
                # Perform RK4 step
                self.sheep_count, self.wolf_count = self.runge_kutta_step(
                    self.sheep_count, 
                    self.wolf_count
                )
                
                # Update average ages (simplified for RK4 - using midpoint estimate)
                if self.sheep_count > 0:
                    # Estimate birth rate for age tracking
                    logistic_growth = 0.0
                    if self.sheep_count > 1 and self.carrying_capacity > 0:
                        logistic_growth = self.sheep_birth_rate * self.sheep_count * (1.0 - self.sheep_count / self.carrying_capacity)
                        logistic_growth = max(0.0, logistic_growth)
                    
                    birth_fraction = (logistic_growth * self.dt) / max(self.sheep_count, 1.0)
                    self.sheep_avg_age = (1.0 - birth_fraction) * (self.sheep_avg_age + self.dt) + birth_fraction * 0.0
                    self.sheep_avg_age = max(0.0, self.sheep_avg_age)
                
                if self.wolf_count > 0:
                    # Estimate birth rate for age tracking
                    predation_rate = self.calculate_predation_rate(self.sheep_count, self.wolf_count)
                    wolf_birth = 0.0
                    if self.wolf_count > 1 and predation_rate > 0:
                        wolf_birth = predation_rate * self.conversion_efficiency
                    
                    birth_fraction = (wolf_birth * self.dt) / max(self.wolf_count, 1.0)
                    self.wolf_avg_age = (1.0 - birth_fraction) * (self.wolf_avg_age + self.dt) + birth_fraction * 0.0
                    self.wolf_avg_age = max(0.0, self.wolf_avg_age)
                
                # Safety check: if population <= 1, set to 0 (extinction - cannot reproduce)
                # Note: This takes precedence over refuge_size - if population drops to 1 or below, extinction occurs
                if self.sheep_count <= 1:
                    self.sheep_count = 0.0
                    self.sheep_avg_age = 0.0
                
                if self.wolf_count <= 1:
                    self.wolf_count = 0.0
                    self.wolf_avg_age = 0.0
                
                # Safety check: wolves die if not eating sheep (starvation)
                predation_rate_check = self.calculate_predation_rate(self.sheep_count, self.wolf_count)
                if predation_rate_check == 0 and self.wolf_count > 0:
                    self.wolf_count = 0.0
                    self.wolf_avg_age = 0.0
                
                # Safety check: ensure sheep never go below refuge size (if refuge exists and population > 1)
                # This only applies if population is above 1 (refuge protects from predation, not from extinction)
                if self.refuge_size > 0 and self.sheep_count > 1:
                    self.sheep_count = max(self.refuge_size, self.sheep_count)
            
            # Yield after completing one year
            yield self.env.timeout(1)
                
    def record_state(self, record_interval=1):
        """Record current state of the ecosystem at specified intervals"""
        while True:
            yield self.env.timeout(record_interval)
            # Round to integers and ensure extinction is properly recorded
            sheep = int(round(self.sheep_count))
            wolves = int(round(self.wolf_count))
            self.history.append({
                'time': self.env.now,
                'sheep': sheep,
                'wolves': wolves
            })

def run_simulation(initial_sheep=100, initial_wolves=20,
                  sheep_birth_rate=0.6, conversion_efficiency=0.2,
                  sheep_lifespan=11.0, wolf_lifespan=13.0,
                  predation_rate=0.1, carrying_capacity=800,
                  refuge_size=10.0,
                  duration=500, disease_factor=0.0, 
                  environmental_stress=0.0, sheep_competition=0.0,
                  wolf_competition=0.0, migration_rate=0.0):
    """Run the simulation using Modified Lotka-Volterra system
    
    Parameters (based on real-world ecological data):
    - sheep_birth_rate: Annual reproduction rate (real: 0.5-0.75 per ewe/year)
    - conversion_efficiency: Efficiency converting eaten sheep into wolf offspring
    - sheep_lifespan: Expected lifespan of sheep in years (real: 10-12 years)
    - wolf_lifespan: Expected lifespan of wolves in years (real: 12-14 years)
    - predation_rate: Combined attack rate and handling time for Holling Type II predation
    - carrying_capacity: Maximum sustainable sheep population
    - refuge_size: Prey refuge - sheep below this count are safe from predation
    - duration: Simulation duration in years
    """
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
    
    # Determine recording interval based on duration
    # For short durations: record every year
    # For medium durations: record every 5 years
    # For long durations: record every 10 years
    if duration <= 50:
        record_interval = 1  # Every year
    elif duration <= 200:
        record_interval = 5  # Every 5 years
    else:
        record_interval = 10  # Every decade
    
    # Calculate start year (2026 - duration)
    start_year = 2026 - duration
    
    # Record initial state at time 0 (will be converted to calendar year later)
    ecosystem.history.append({
        'time': 0.0,  # Simulation time, will convert to calendar year
        'sheep': int(round(initial_sheep)),
        'wolves': int(round(initial_wolves))
    })
    
    # Start synchronized population update process
    env.process(ecosystem.update_populations())
    env.process(ecosystem.record_state(record_interval))
    
    # Run simulation
    env.run(until=duration)
    
    # Convert time values to calendar years
    for entry in ecosystem.history:
        # entry['time'] is currently simulation time (0 to duration)
        # Convert to calendar year
        sim_time = entry['time']
        calendar_year = start_year + sim_time
        entry['time'] = int(round(calendar_year))
    
    return ecosystem.history

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json
    
    # Extract parameters with defaults (updated to realistic values based on ecological studies)
    initial_sheep = int(data.get('initial_sheep', 100))
    initial_wolves = int(data.get('initial_wolves', 20))
    sheep_birth_rate = float(data.get('sheep_birth_rate', 0.6))  # Real: 0.5-0.75 per ewe/year
    conversion_efficiency = float(data.get('conversion_efficiency', 0.2))  # Real: ~0.03-0.05
    sheep_lifespan = float(data.get('sheep_lifespan', 11.0))  # Real: 10-12 years
    wolf_lifespan = float(data.get('wolf_lifespan', 13.0))  # Real: 12-14 years
    predation_rate = float(data.get('predation_rate', 0.1))  # Combined attack rate and handling time
    carrying_capacity = int(data.get('carrying_capacity', 800))  # Optimized for oscillations
    refuge_size = float(data.get('refuge_size', 10.0))  # Prey refuge safety buffer
    disease_factor = float(data.get('disease_factor', 0.0))
    environmental_stress = float(data.get('environmental_stress', 0.0))
    sheep_competition = float(data.get('sheep_competition', 0.0))
    wolf_competition = float(data.get('wolf_competition', 0.0))
    migration_rate = float(data.get('migration_rate', 0.0))
    duration = int(data.get('duration', 500))
    num_runs = int(data.get('num_runs', 1))
    
    # Run multiple simulations and average results
    all_runs = []
    for _ in range(num_runs):
        results = run_simulation(
            initial_sheep=initial_sheep,
            initial_wolves=initial_wolves,
            sheep_birth_rate=sheep_birth_rate,
            conversion_efficiency=conversion_efficiency,
            sheep_lifespan=sheep_lifespan,
            wolf_lifespan=wolf_lifespan,
            predation_rate=predation_rate,
            carrying_capacity=carrying_capacity,
            refuge_size=refuge_size,
            duration=duration,
            disease_factor=disease_factor,
            environmental_stress=environmental_stress,
            sheep_competition=sheep_competition,
            wolf_competition=wolf_competition,
            migration_rate=migration_rate
        )
        all_runs.append(results)
    
    # Average across runs (round to nearest integer)
    if num_runs == 1:
        averaged_results = all_runs[0]
    else:
        # Initialize averaged results - ensure we have the same time points
        averaged_results = []
        # Use the first run's time structure as reference
        reference_times = [entry['time'] for entry in all_runs[0]]
        
        for i, ref_time in enumerate(reference_times):
            avg_sheep = sum(run[i]['sheep'] for run in all_runs) / num_runs
            avg_wolves = sum(run[i]['wolves'] for run in all_runs) / num_runs
            averaged_results.append({
                'time': int(ref_time),  # Preserve calendar year
                'sheep': int(round(avg_sheep)),
                'wolves': int(round(avg_wolves))
            })
    
    return jsonify(averaged_results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

