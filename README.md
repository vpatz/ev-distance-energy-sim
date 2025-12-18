# Physics-based Synthetic Data Generator for  Electric Vehicle Range Prediction  
Simulate battery power dissipation for electric vehicles to generate synthetic data for drive scenarios.

## Data format
 

| Column names | Description |
|--------------|-------------|
| time_stamp_sec | Time stamp for end of window (sec) |
| window_duration_sec | Duration of time window (sec) |
| avg_speed_kmh | Averaged speed of vehicle in the time window (km/h) | 
| grade | Incline of the road as ratio |
| vehicle_mass | Weight of vehicle (kg) |
| battery_energy_spend_kwh | Total energy spent from battery in the time window (kWh) |
| motion_energy_drain_kwh | Energy drained for horizontal vehicle motion due to rolling resistance (kWh) |
| gravity_energy_drain_kwh | Energy drained for overcoming gravity on inclines (kWh) |
| heat_energy_drain_kwh | Energy loss due to battery heat dissipation (kWh) |
| aux_energy_drain_kwh | Auxiliary energy consumption by HVAC, HMI, and body control ECUs (kWh) |
| regen_energy_gain_kwh | Energy returned by regenerative braking (kWh) |
| soc_percent | State of charge of the battery (%) |
| distance_km | Distance traveled in the time window (km) |
| remaining_range_km | Estimated remaining range based on current SOC and energy consumption rate (km) | 

## Application for training distance prediction models
- Evaluating federated learning of Range Prediction Models
- Validating physical constants for energy efficiency
- Can be modified for other physical processes that involve energy transformations 
