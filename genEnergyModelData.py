# create multi-dim time series data using simplified energy equation

# In a time window, energy from the battery is used for 
# 1 - horizontal motion
# 2 - climbing on inclines (or descending)
# 3 - battery heat dissipation 
# 4 - aux enery consumption by HVAC, HMI, body control ECUs
# and returned from regenerative braking 

# Therefore simplified energy equation is (ignoring wind resistance)
# for a time window T, vehicle travelling at avg. velocity V
# E_batt = E_motion + E_grav + E_heat + E_aux - E_regen
# where 
# E_batt    -> battery energy drain 
# E_motion  -> horizontal motion energy 
# E_grav    -> energy for overcoming gravity (on inclines)
# E_heat    -> energy loss by battery heating
# E_aux     -> aux energy consumption
# E_regen   -> energy returned by regenerative braking

# E_motion = F_motion x distance
#          = u_roll * M * g * cos(theta) * distance 
# where
# u_roll    -> rolling friction coefficient
# M         -> mass of the vehicle 
# g         -> acceleration due to gravity 
# theta     -> angle of incline
# distance  -> V * T

# E_grav = M * g * sin(theta) * distance

# E_heat    -> battery energy loss due to heating


# scenarios
# 1. Level-road driving with constant speed and no regenerative braking
# 2. Level-road driving with varying speed and regenerative braking
# 3. Uphill-road driving with no regenerative braking
# 4. Downhill-road driving with regenerative braking

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch


# =============================================================================
# Physical Constants and Vehicle Parameters
# =============================================================================

@dataclass
class VehicleParams:
    """EV vehicle parameters for energy calculation"""
    mass: float = 1500.0                # Vehicle mass (kg)
    battery_capacity: float = 150.0     # Battery capacity (kWh)
    #frontal_area: float = 2.5          # Frontal area (m^2)
    #drag_coeff: float = 0.28           # Aerodynamic drag coefficient
    #rolling_resistance: float = 0.012  # Rolling resistance coefficient
    u_roll: float = 0.002               # Rolling resistance coefficient
    #K_m: float = 0.75                  # Motor efficiency (battery to kinetic)
    K_regen: float = 0.2                # Ratio of translation motion energy that is regenerated
    P_aux: float = 0.05                  # Auxiliary systems power consumption (kW)
    P_heat: float = 0.01                 # Battery thermal loss (kW)




@dataclass
class EnvironmentParams:
    """Environmental parameters"""
    #air_density: float = 1.225   # Air density (kg/m^3)
    gravity: float = 9.81        # Gravitational acceleration (m/s^2)
    #ambient_temp: float = 25.0   # Ambient temperature (°C)

@dataclass
class TimeWindowParams:
    time: float = 15 # duration of time window (secs)
    speed: float = 40 # average speed in (km/h)
    grade: float = 0 # flat road default

# =============================================================================
# Energy Calculation Functions
# =============================================================================

'''def calculate_aerodynamic_drag(
    velocity: float, 
    vehicle: VehicleParams, 
    env: EnvironmentParams
) -> float:
    """
    Calculate aerodynamic drag force.
    F_drag = 0.5 * rho * Cd * A * v^2
    """
    return 0.5 * env.air_density * vehicle.drag_coeff * vehicle.frontal_area * velocity**2
'''

def calculate_E_motion(
    vehicle: VehicleParams, 
    env: EnvironmentParams, 
    timeWindow: TimeWindowParams,
    grade: float = 0.0
) -> float:
    """
    Calculate rolling resistance force.
    F_roll = Crr * m * g * cos(theta)
    E_roll = F_roll * time
    """
    theta = np.arctan(grade / 100)  # Convert grade percentage to angle
    surface_distance = (timeWindow.speed * 1000.0/3600.0) * timeWindow.time # convert speed from km/hr to m/s
    
    # E_motion = u_roll * M * g * cos(theta) * distance 
 
    return vehicle.u_roll * vehicle.mass * env.gravity * np.cos(theta) * surface_distance / 1000.0 /60.0 # convert to kW hour


def calculate_E_grav(
    vehicle: VehicleParams, 
    env: EnvironmentParams, 
    timeWindow: TimeWindowParams,
    grade: float = 0.0
) -> float:
    """
    Calculate gravitational force due to road grade.
    F_grade = m * g * sin(theta)
    E_grav = F_grade * distance * sin(theta)
    Positive for uphill, negative for downhill
    """
    theta = np.arctan(grade / 100)
    vert_distance = (timeWindow.speed * 1000.0/3600.0) * timeWindow.time * np.sin(theta)
    return vehicle.mass * env.gravity * np.sin(theta) * vert_distance / 1000.0 / 60.0 # convert to kW hour



def calculate_energy_loss_in_time_window(
    #speed: float,
    #acceleration: float,
    #grade: float,
    vehicle: VehicleParams,
    env: EnvironmentParams,
    timeWindow: TimeWindowParams
) -> float:
    
    
    
    E_motion = calculate_E_motion(vehicle, env, timeWindow)
    E_grav = calculate_E_grav(vehicle, env, timeWindow)
    E_heat = vehicle.P_heat * timeWindow.time
    E_aux = vehicle.P_aux * timeWindow.time

    E_regen = vehicle.K_regen *  E_motion

    E_total_loss = E_motion + E_grav + E_heat + E_aux - E_regen

       
    return E_total_loss, E_motion, E_grav, E_heat, E_aux, E_regen


'''def calculate_power_consumption(
    velocity: float,
    acceleration: float,
    grade: float,
    vehicle: VehicleParams,
    env: EnvironmentParams
) -> Tuple[float, float]:
    """
    Calculate power consumption/regeneration.
    Returns (power_consumed, energy_regenerated) in kW
    """
    total_force = calculate_total_force(velocity, acceleration, grade, vehicle, env)
    mechanical_power = total_force * velocity / 1000  # Convert to kW
    
    if mechanical_power >= 0:
        # Positive power needed (accelerating, uphill, or overcoming resistance)
        battery_power = mechanical_power / vehicle.K_m + vehicle.aux_power
        regen_power = 0.0
    else:
        # Negative power (decelerating or downhill) - regenerative braking
        regen_power = abs(mechanical_power) * vehicle.K_r
        battery_power = vehicle.aux_power
    
    return battery_power, regen_power'''


'''def calculate_heat_loss(
    battery_power: float,
    battery_temp: float,
    ambient_temp: float,
    vehicle: VehicleParams
) -> float:
    """
    Calculate battery energy loss due to heat.
    Simplified model: higher temperature difference = more losses
    """
    temp_factor = 1.0 + 0.005 * abs(battery_temp - 25.0)  # Baseline at 25°C
    internal_resistance_loss = battery_power * (1 - vehicle.K_m) * temp_factor
    return max(0, internal_resistance_loss)'''


# =============================================================================
# Scenario Generation Functions
# =============================================================================

def generate_scenario_data(
    drive_duration_sec: int = 36000,
    window_size_sec: int = 15,
    speed_kmph: float = 40.0,
    initial_soc_percent: float = 95.0,
    grade: Optional[float] = 0,
    vehicle: Optional[VehicleParams] = None,
    env: Optional[EnvironmentParams] = None
) -> pd.DataFrame:
    """
    Scenario 1: Level-road driving with constant speed and no regenerative braking
    
    Args:
        duration: Trip duration in seconds
        dt: Time step in seconds (default: 900 = 15 minutes)
        speed_kmh: Constant speed in km/h
        initial_soc: Initial state of charge (%)
        vehicle: Vehicle parameters
        env: Environment parameters
    
    Returns:
        DataFrame with time series data
    """
    vehicle = vehicle or VehicleParams()
    env = env or EnvironmentParams()
    timeWindow = TimeWindowParams()
    timeWindow.speed = speed_kmph
    timeWindow.time = window_size_sec

    
    #velocity = speed_kmph / 3.6  # Convert to m/s
    time_steps = int(drive_duration_sec / window_size_sec)
    
    # Initialize arrays
    time_sec = np.arange(0, drive_duration_sec, window_size_sec)
    data = {
        'time_stamp_sec': time_sec,
        'window_duration_sec': np.full(time_steps, window_size_sec),
        'avg_speed_kmh': np.full(time_steps, speed_kmph),
    #    'acceleration': np.zeros(time_steps),
        #'grade': np.zeros(time_steps),
        'grade': np.full(time_steps, grade),
        'vehicle_mass': np.full(time_steps, vehicle.mass),
        'battery_energy_spend_kwh': np.zeros(time_steps),
        'motion_energy_drain_kwh': np.zeros(time_steps),
        'gravity_energy_drain_kwh': np.zeros(time_steps),
        'heat_energy_drain_kwh': np.zeros(time_steps),
        'aux_energy_drain_kwh': np.zeros(time_steps),
        'regen_energy_gain_kwh': np.zeros(time_steps),
        'soc_percent': np.zeros(time_steps),
        #'battery_temp': np.zeros(time_steps),
        'distance_km': np.zeros(time_steps),
        #'energy_consumed_kwh': np.zeros(time_steps),
        
        'remaining_range_km': np.zeros(time_steps),
    }
    
    # Initial conditions
    current_soc_percent = initial_soc_percent
    #battery_temp = env.ambient_temp + 5.0  # Battery slightly warmer than ambient
    total_distance_km = 0.0 
    #total_energy = 0.0
    
    for i in range(time_steps):

        if current_soc_percent <= 1: # stop when soc is < 1%
            break


        # Calculate energy spent from battery in time window
        step_energy_spend, motion_energy_drain, gravity_energy_drain, heat_energy_drain, aux_energy_drain, regen_energy_gain  = calculate_energy_loss_in_time_window(
            vehicle, env, timeWindow
        )
        
        # Update energy and SOC
        #energy_step = (battery_power + heat_loss) * dt / 3600  # kWh
        #total_energy += energy_step
        current_soc_percent -= (step_energy_spend / vehicle.battery_capacity) * 100
        
        # Update distance
        distance_step_km = speed_kmph * (window_size_sec /3600.0) 
        total_distance_km += distance_step_km
        
        # Update battery temperature (simplified)
        #battery_temp += (battery_power * 0.01 - 0.05 * (battery_temp - env.ambient_temp)) * dt / 60
        
        # Calculate remaining range
        if current_soc_percent > 0:
            energy_spend_per_km = step_energy_spend / distance_step_km   # kWh/km
            remaining_energy = (current_soc_percent / 100) * vehicle.battery_capacity
            remaining_range_km = remaining_energy / energy_spend_per_km
            #if energy_rate > 0 else 0
        else:
            remaining_range_km = 0
        
        # Store data
        data['battery_energy_spend_kwh'][i] = step_energy_spend
        data['motion_energy_drain_kwh'][i] = motion_energy_drain
        data['gravity_energy_drain_kwh'][i] = gravity_energy_drain
        data['heat_energy_drain_kwh'][i] = heat_energy_drain
        data['aux_energy_drain_kwh'][i] = aux_energy_drain
        data['regen_energy_gain_kwh'][i] = regen_energy_gain
        data['soc_percent'][i] = current_soc_percent
        #data['battery_temp'][i] = battery_temp
        data['distance_km'][i] = distance_step_km
        #data['energy_consumed_kwh'][i] = total_energy
        data['remaining_range_km'][i] = remaining_range_km
    
    return pd.DataFrame(data)


def generate_scenario_2_level_varying(
    duration: int = 3600,
    dt: float = 900.0,
    initial_soc: float = 90.0,
    vehicle: Optional[VehicleParams] = None,
    env: Optional[EnvironmentParams] = None
) -> pd.DataFrame:
    """
    Scenario 2: Level-road driving with varying speed and regenerative braking
    
    Simulates city/suburban driving with speed variations and stops
    
    Args:
        duration: Trip duration in seconds
        dt: Time step in seconds (default: 900 = 15 minutes)
    """
    vehicle = vehicle or VehicleParams()
    env = env or EnvironmentParams()
    
    time_steps = int(duration / dt)
    
    # Generate realistic speed profile with accelerations and decelerations
    speed_profile = generate_urban_speed_profile(duration, dt)
    
    # Initialize arrays
    time_seconds = np.arange(0, duration, dt)
    data = {
        'time_seconds': time_seconds,
        'time_minutes': time_seconds / 60,
        'speed_kmh': speed_profile,
        'acceleration': np.zeros(time_steps),
        'grade': np.zeros(time_steps),
        'battery_power_kw': np.zeros(time_steps),
        'regen_power_kw': np.zeros(time_steps),
        'soc': np.zeros(time_steps),
        'battery_temp': np.zeros(time_steps),
        'distance_km': np.zeros(time_steps),
        'energy_consumed_kwh': np.zeros(time_steps),
        'remaining_range_km': np.zeros(time_steps),
    }
    
    # Calculate accelerations from speed profile
    for i in range(1, time_steps):
        data['acceleration'][i] = (speed_profile[i] - speed_profile[i-1]) / (3.6 * dt)
    
    # Initial conditions
    current_soc = initial_soc
    battery_temp = env.ambient_temp + 5.0
    total_distance = 0.0
    total_energy = 0.0
    
    for i in range(time_steps):
        velocity = speed_profile[i] / 3.6  # m/s
        acceleration = data['acceleration'][i]
        
        # Calculate power consumption/regeneration
        battery_power, regen_power = calculate_power_consumption(
            velocity, acceleration, 0.0, vehicle, env
        )
        
        # Apply regenerative braking energy back to battery
        net_power = battery_power - regen_power
        
        # Calculate heat loss
        heat_loss = calculate_heat_loss(battery_power, battery_temp, env.ambient_temp, vehicle)
        
        # Update energy and SOC
        energy_step = (net_power + heat_loss) * dt / 3600
        total_energy += max(0, energy_step)
        current_soc -= (energy_step / vehicle.battery_capacity) * 100
        
        # Update distance
        distance_step = velocity * dt / 1000
        total_distance += distance_step
        
        # Update battery temperature
        battery_temp += (abs(net_power) * 0.01 - 0.05 * (battery_temp - env.ambient_temp)) * dt / 60
        
        # Calculate remaining range
        if total_energy > 0 and total_distance > 0:
            avg_consumption = total_energy / total_distance  # kWh/km
            remaining_energy = (current_soc / 100) * vehicle.battery_capacity
            remaining_range = remaining_energy / avg_consumption if avg_consumption > 0 else 0
        else:
            remaining_range = (current_soc / 100) * vehicle.battery_capacity * 5  # Rough estimate
        
        # Store data
        data['battery_power_kw'][i] = battery_power
        data['regen_power_kw'][i] = regen_power
        data['soc'][i] = max(0, min(100, current_soc))
        data['battery_temp'][i] = battery_temp
        data['distance_km'][i] = total_distance
        data['energy_consumed_kwh'][i] = total_energy
        data['remaining_range_km'][i] = max(0, remaining_range)
    
    return pd.DataFrame(data)


def generate_scenario_3_uphill(
    duration: int = 1800,
    dt: float = 900.0,
    speed_kmh: float = 60.0,
    grade_percent: float = 5.0,
    initial_soc: float = 90.0,
    vehicle: Optional[VehicleParams] = None,
    env: Optional[EnvironmentParams] = None
) -> pd.DataFrame:
    """
    Scenario 3: Uphill-road driving with no regenerative braking
    
    Args:
        duration: Trip duration in seconds
        dt: Time step in seconds (default: 900 = 15 minutes)
        grade_percent: Road grade in percentage (positive for uphill)
    """
    vehicle = vehicle or VehicleParams()
    env = env or EnvironmentParams()
    
    velocity = speed_kmh / 3.6
    time_steps = int(duration / dt)
    
    # Initialize arrays
    time_seconds = np.arange(0, duration, dt)
    data = {
        'time_seconds': time_seconds,
        'time_minutes': time_seconds / 60,
        'speed_kmh': np.full(time_steps, speed_kmh),
        'acceleration': np.zeros(time_steps),
        'grade': np.full(time_steps, grade_percent),
        'battery_power_kw': np.zeros(time_steps),
        'regen_power_kw': np.zeros(time_steps),
        'soc': np.zeros(time_steps),
        'battery_temp': np.zeros(time_steps),
        'distance_km': np.zeros(time_steps),
        'energy_consumed_kwh': np.zeros(time_steps),
        'remaining_range_km': np.zeros(time_steps),
    }
    
    current_soc = initial_soc
    battery_temp = env.ambient_temp + 5.0
    total_distance = 0.0
    total_energy = 0.0
    
    for i in range(time_steps):
        # Calculate power consumption (uphill, no regen)
        battery_power, regen_power = calculate_power_consumption(
            velocity, 0.0, grade_percent, vehicle, env
        )
        
        # No regenerative braking on uphill
        regen_power = 0.0
        
        heat_loss = calculate_heat_loss(battery_power, battery_temp, env.ambient_temp, vehicle)
        
        energy_step = (battery_power + heat_loss) * dt / 3600
        total_energy += energy_step
        current_soc -= (energy_step / vehicle.battery_capacity) * 100
        
        distance_step = velocity * dt / 1000
        total_distance += distance_step
        
        # Higher temperature rise due to higher power demand
        battery_temp += (battery_power * 0.015 - 0.05 * (battery_temp - env.ambient_temp)) * dt / 60
        
        # Calculate remaining range considering uphill consumption
        if battery_power > 0:
            energy_rate = battery_power / velocity * 1000
            remaining_energy = (current_soc / 100) * vehicle.battery_capacity
            remaining_range = remaining_energy / energy_rate if energy_rate > 0 else 0
        else:
            remaining_range = 0
        
        data['battery_power_kw'][i] = battery_power
        data['regen_power_kw'][i] = regen_power
        data['soc'][i] = max(0, current_soc)
        data['battery_temp'][i] = battery_temp
        data['distance_km'][i] = total_distance
        data['energy_consumed_kwh'][i] = total_energy
        data['remaining_range_km'][i] = max(0, remaining_range)
    
    return pd.DataFrame(data)


def generate_scenario_4_downhill(
    duration: int = 1800,
    dt: float = 900.0,
    speed_kmh: float = 60.0,
    grade_percent: float = -5.0,
    initial_soc: float = 70.0,
    vehicle: Optional[VehicleParams] = None,
    env: Optional[EnvironmentParams] = None
) -> pd.DataFrame:
    """
    Scenario 4: Downhill-road driving with regenerative braking
    
    Args:
        duration: Trip duration in seconds
        dt: Time step in seconds (default: 900 = 15 minutes)
        grade_percent: Road grade in percentage (negative for downhill)
    """
    vehicle = vehicle or VehicleParams()
    env = env or EnvironmentParams()
    
    velocity = speed_kmh / 3.6
    time_steps = int(duration / dt)
    
    time_seconds = np.arange(0, duration, dt)
    data = {
        'time_seconds': time_seconds,
        'time_minutes': time_seconds / 60,
        'speed_kmh': np.full(time_steps, speed_kmh),
        'acceleration': np.zeros(time_steps),
        'grade': np.full(time_steps, grade_percent),
        'battery_power_kw': np.zeros(time_steps),
        'regen_power_kw': np.zeros(time_steps),
        'soc': np.zeros(time_steps),
        'battery_temp': np.zeros(time_steps),
        'distance_km': np.zeros(time_steps),
        'energy_consumed_kwh': np.zeros(time_steps),
        'remaining_range_km': np.zeros(time_steps),
    }
    
    current_soc = initial_soc
    battery_temp = env.ambient_temp + 5.0
    total_distance = 0.0
    total_energy_consumed = 0.0
    total_energy_regenerated = 0.0
    
    for i in range(time_steps):
        # Calculate power (downhill produces regeneration)
        battery_power, regen_power = calculate_power_consumption(
            velocity, 0.0, grade_percent, vehicle, env
        )
        
        heat_loss = calculate_heat_loss(battery_power, battery_temp, env.ambient_temp, vehicle)
        
        # Net energy considering regeneration
        net_power = battery_power - regen_power
        energy_step = (net_power + heat_loss) * dt / 3600
        
        if energy_step > 0:
            total_energy_consumed += energy_step
        else:
            total_energy_regenerated += abs(energy_step)
        
        current_soc -= (energy_step / vehicle.battery_capacity) * 100
        current_soc = min(100, current_soc)  # Cap at 100%
        
        distance_step = velocity * dt / 1000
        total_distance += distance_step
        
        battery_temp += (abs(net_power) * 0.008 - 0.05 * (battery_temp - env.ambient_temp)) * dt / 60
        
        # Calculate remaining range
        net_energy = total_energy_consumed - total_energy_regenerated
        if net_energy > 0 and total_distance > 0:
            avg_consumption = net_energy / total_distance
            remaining_energy = (current_soc / 100) * vehicle.battery_capacity
            remaining_range = remaining_energy / avg_consumption if avg_consumption > 0 else 0
        else:
            remaining_range = (current_soc / 100) * vehicle.battery_capacity * 6
        
        data['battery_power_kw'][i] = battery_power
        data['regen_power_kw'][i] = regen_power
        data['soc'][i] = max(0, min(100, current_soc))
        data['battery_temp'][i] = battery_temp
        data['distance_km'][i] = total_distance
        data['energy_consumed_kwh'][i] = total_energy_consumed
        data['remaining_range_km'][i] = max(0, remaining_range)
    
    return pd.DataFrame(data)


# =============================================================================
# Helper Functions
# =============================================================================

def generate_urban_speed_profile(duration: int, dt: float) -> np.ndarray:
    """
    Generate a realistic urban driving speed profile with stops and speed variations.
    
    For 15-minute intervals, each value represents the average speed during that period.
    
    Args:
        duration: Total duration in seconds
        dt: Time step in seconds (default: 900 = 15 minutes)
    
    Returns:
        Array of speeds in km/h
    """
    time_steps = int(duration / dt)
    speed_profile = np.zeros(time_steps)
    
    # For 15-minute intervals, we model average speed over the period
    # Urban driving averages 25-45 km/h with traffic, highway 60-80 km/h
    
    # Base average speed for urban/suburban driving
    base_speed = np.random.uniform(30, 50)
    
    for i in range(time_steps):
        # Simulate varying traffic conditions over 15-min periods
        traffic_factor = np.random.choice(
            ['light', 'moderate', 'heavy', 'congested'],
            p=[0.3, 0.4, 0.2, 0.1]
        )
        
        if traffic_factor == 'light':
            avg_speed = base_speed + np.random.uniform(10, 20)
        elif traffic_factor == 'moderate':
            avg_speed = base_speed + np.random.uniform(-5, 10)
        elif traffic_factor == 'heavy':
            avg_speed = base_speed + np.random.uniform(-15, 0)
        else:  # congested
            avg_speed = base_speed + np.random.uniform(-25, -10)
        
        # Clamp speed between 10 and 80 km/h (average over 15 min won't be 0)
        avg_speed = max(10, min(80, avg_speed))
        speed_profile[i] = avg_speed
    
    return speed_profile


def convert_to_model_features(df: pd.DataFrame, vehicle: VehicleParams) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert scenario DataFrame to model input features and target.
    
    Features (7 dimensions):
    1. State of Charge (SOC) (%)
    2. Battery Voltage (V) - estimated from SOC
    3. Battery Temperature (°C)
    4. Current Vehicle Speed (km/h)
    5. Average Speed over the last 5 minutes
    6. Current draw (A) - estimated from power
    7. State of Health (SOH) (%) - assumed constant
    
    Target: Remaining range (km)
    """
    n_samples = len(df)
    
    # Calculate rolling average speed (5 minute window = 300 seconds)
    window_size = min(300, n_samples)
    avg_speed = df['speed_kmh'].rolling(window=window_size, min_periods=1).mean()
    
    # Estimate battery voltage from SOC (linear approximation: 320V at 0%, 400V at 100%)
    battery_voltage = 320 + (df['soc'] / 100) * 80
    
    # Estimate current draw from power (P = V * I)
    current_draw = df['battery_power_kw'] * 1000 / battery_voltage
    
    # State of Health (assuming 95% for generated data)
    soh = np.full(n_samples, 95.0)
    
    # Build feature matrix
    features = np.column_stack([
        df['soc'].values,
        battery_voltage.values,
        df['battery_temp'].values,
        df['speed_kmh'].values,
        avg_speed.values,
        current_draw.values,
        soh
    ])
    
    # Target: remaining range
    targets = df['remaining_range_km'].values.reshape(-1, 1)
    
    return torch.FloatTensor(features), torch.FloatTensor(targets)


def generate_mixed_scenario_dataset(
    num_samples_per_scenario: int = 2500,
    vehicle: Optional[VehicleParams] = None,
    env: Optional[EnvironmentParams] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a mixed dataset from all scenarios for model training.
    
    Returns:
        Tuple of (features, targets) tensors
    """
    vehicle = vehicle or VehicleParams()
    env = env or EnvironmentParams()
    
    all_features = []
    all_targets = []
    
    scenarios = [
        ('level_constant', generate_scenario_1_level_constant),
        ('level_varying', generate_scenario_2_level_varying),
        ('uphill', generate_scenario_3_uphill),
        ('downhill', generate_scenario_4_downhill),
    ]
    
    samples_collected = 0
    target_per_scenario = num_samples_per_scenario
    
    # Time step for 15-minute intervals
    dt = 900  # 15 minutes in seconds
    
    for name, scenario_func in scenarios:
        scenario_samples = 0
        iteration = 0
        
        while scenario_samples < target_per_scenario:
            # Vary initial conditions for diversity
            initial_soc = np.random.uniform(30, 95)
            
            if name == 'level_constant':
                speed = np.random.uniform(50, 120)
                # Duration must be multiple of dt (15 min), range: 30 min to 4 hours
                duration = np.random.choice([1800, 2700, 3600, 5400, 7200, 10800, 14400])
                df = scenario_func(duration=duration, speed_kmh=speed, initial_soc=initial_soc, vehicle=vehicle, env=env)
            elif name == 'level_varying':
                # Duration: 30 min to 2 hours
                duration = np.random.choice([1800, 2700, 3600, 5400, 7200])
                df = scenario_func(duration=duration, initial_soc=initial_soc, vehicle=vehicle, env=env)
            elif name == 'uphill':
                grade = np.random.uniform(2, 10)
                speed = np.random.uniform(40, 80)
                # Duration: 15 min to 1 hour
                duration = np.random.choice([900, 1800, 2700, 3600])
                df = scenario_func(duration=duration, speed_kmh=speed, grade_percent=grade, initial_soc=initial_soc, vehicle=vehicle, env=env)
            else:  # downhill
                grade = np.random.uniform(-10, -2)
                speed = np.random.uniform(40, 80)
                # Duration: 15 min to 1 hour
                duration = np.random.choice([900, 1800, 2700, 3600])
                df = scenario_func(duration=duration, speed_kmh=speed, grade_percent=grade, initial_soc=initial_soc, vehicle=vehicle, env=env)
            
            features, targets = convert_to_model_features(df, vehicle)
            
            # Subsample to avoid too much correlated data
            indices = np.random.choice(len(features), size=min(100, len(features)), replace=False)
            all_features.append(features[indices])
            all_targets.append(targets[indices])
            
            scenario_samples += len(indices)
            iteration += 1
            
            if iteration > 100:  # Safety limit
                break
        
        print(f"Generated {scenario_samples} samples for scenario: {name}")
    
    # Concatenate all samples
    X = torch.cat(all_features, dim=0)
    y = torch.cat(all_targets, dim=0)
    
    # Shuffle
    perm = torch.randperm(len(X))
    X = X[perm]
    y = y[perm]
    
    print(f"\nTotal dataset: {len(X)} samples")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def save_dataset(X: torch.Tensor, y: torch.Tensor, filename: str = 'ev_dataset.pt'):
    """Save the dataset to a file"""
    torch.save({'features': X, 'targets': y}, filename)
    print(f"Dataset saved to {filename}")


def load_dataset(filename: str = 'ev_dataset.pt') -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a saved dataset"""
    data = torch.load(filename)
    return data['features'], data['targets']


def save_scenario_to_csv(df: pd.DataFrame, scenario_name: str, output_dir: str = 'scenario_data') -> str:
    """
    Save scenario DataFrame to CSV file.
    
    Args:
        df: DataFrame containing scenario data
        scenario_name: Name of the scenario (used for filename)
        output_dir: Directory to save CSV files
    
    Returns:
        Path to the saved CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    filename = f"{scenario_name}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"  Saved to: {filepath}")
    
    return filepath


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to generate and demonstrate EV energy model data"""
    
    print("=" * 70)
    print("EV Range Prediction - Data Generation")
    print("=" * 70)
    
    # Initialize parameters
    vehicle = VehicleParams()
    env = EnvironmentParams()
    drive_duration_sec = 3600 # sec
    window_size_sec = 30 # sec
    speed_kmph = 80 # km / h
    initial_soc_percent = 90 # percent of vehicle batter capacity 
    
    print("\nVehicle Parameters:")
    print(f"  Mass: {vehicle.mass} kg")
    print(f"  Battery Capacity: {vehicle.battery_capacity} kWh")
    #print(f"  Motor Efficiency (K_m): {vehicle.K_m}")
    #print(f"  Regen Efficiency (K_r): {vehicle.K_r}")
    
    # Generate individual scenario demonstrations
    print("\n" + "=" * 70)
    print("Generating Individual Scenarios...")
    print("=" * 70)
    
    # Scenario 1: Level road, constant speed
    print("\n[Scenario 1] Level road, constant speed (80 km/h)...")
    df1 = generate_scenario_data(drive_duration_sec, window_size_sec, speed_kmph, initial_soc_percent)
    #print(f"  Duration: 600 minutes")
    print(f"  Final SOC: {df1['soc_percent'].iloc[-1]:.1f}%")
    print(f"  Distance traveled: {df1['distance_km'].iloc[-1]:.1f} km")
    #print(f"  Energy consumed: {df1['energy_consumed_kwh'].iloc[-1]:.2f} kWh")
    save_scenario_to_csv(df1, "scenario_1_level_constant_speed")


"""

    # Scenario 2: Level road, varying speed
    print("\n[Scenario 2] Level road, varying speed (urban driving)...")
    df2 = generate_scenario_2_level_varying(duration=1800, initial_soc=90)
    print(f"  Duration: 30 minutes")
    print(f"  Final SOC: {df2['soc'].iloc[-1]:.1f}%")
    print(f"  Distance traveled: {df2['distance_km'].iloc[-1]:.1f} km")
    print(f"  Energy consumed: {df2['energy_consumed_kwh'].iloc[-1]:.2f} kWh")
    print(f"  Max regen power: {df2['regen_power_kw'].max():.2f} kW")
    save_scenario_to_csv(df2, "scenario_2_level_varying_speed")
    
    # Scenario 3: Uphill
    print("\n[Scenario 3] Uphill driving (5% grade, 60 km/h)...")
    df3 = generate_scenario_3_uphill(duration=900, speed_kmh=60, grade_percent=5, initial_soc=90)
    print(f"  Duration: 15 minutes")
    print(f"  Final SOC: {df3['soc'].iloc[-1]:.1f}%")
    print(f"  Distance traveled: {df3['distance_km'].iloc[-1]:.1f} km")
    print(f"  Energy consumed: {df3['energy_consumed_kwh'].iloc[-1]:.2f} kWh")
    print(f"  Avg power demand: {df3['battery_power_kw'].mean():.2f} kW")
    save_scenario_to_csv(df3, "scenario_3_uphill_driving")
    
    # Scenario 4: Downhill
    print("\n[Scenario 4] Downhill driving (-5% grade, 60 km/h)...")
    df4 = generate_scenario_4_downhill(duration=900, speed_kmh=60, grade_percent=-5, initial_soc=70)
    print(f"  Duration: 15 minutes")
    print(f"  Final SOC: {df4['soc'].iloc[-1]:.1f}%")
    print(f"  Distance traveled: {df4['distance_km'].iloc[-1]:.1f} km")
    print(f"  Energy regenerated via regen braking: {df4['regen_power_kw'].sum() * 1/3600:.2f} kWh")
    save_scenario_to_csv(df4, "scenario_4_downhill_driving")
    
    # Generate full training dataset
    print("\n" + "=" * 70)
    print("Generating Mixed Training Dataset...")
    print("=" * 70)
    
    X, y = generate_mixed_scenario_dataset(num_samples_per_scenario=2500)
    
    # Feature statistics
    print("\nFeature Statistics:")
    feature_names = ['SOC (%)', 'Voltage (V)', 'Temp (°C)', 'Speed (km/h)', 
                     'Avg Speed (km/h)', 'Current (A)', 'SOH (%)']
    for i, name in enumerate(feature_names):
        print(f"  {name}: min={X[:, i].min():.2f}, max={X[:, i].max():.2f}, mean={X[:, i].mean():.2f}")
    
    print(f"\nTarget (Remaining Range) Statistics:")
    print(f"  Min: {y.min():.2f} km")
    print(f"  Max: {y.max():.2f} km")
    print(f"  Mean: {y.mean():.2f} km")
    
    # Save dataset
    save_dataset(X, y, 'ev_training_dataset.pt')
    
    print("\n" + "=" * 70)
    print("Data generation complete!")
    print("=" * 70)
    print("\nOutput files:")
    print("  - scenario_data/scenario_1_level_constant_speed.csv")
    print("  - scenario_data/scenario_2_level_varying_speed.csv")
    print("  - scenario_data/scenario_3_uphill_driving.csv")
    print("  - scenario_data/scenario_4_downhill_driving.csv")
    print("  - ev_training_dataset.pt (PyTorch tensor dataset)")
    
    return X, y
"""


if __name__ == "__main__":
    main()
