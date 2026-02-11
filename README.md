# Physics-based Synthetic Data Generator for Electric Vehicle Range Prediction  
Simulate battery power dissipation for electric vehicles to generate synthetic data for drive scenarios.

## Usage

```bash
python3 genEnergyModelData.py
```

This generates:
- `scenario_data/sample_scenario.csv` - Single scenario demonstration
- `training_data/ev_mixed_training_data.csv` - Full training dataset with all columns
- `training_data/ev_ml_dataset.csv` - ML-ready dataset with features and targets

## Data Format

### Full Dataset Columns

| Column names | Description |
|--------------|-------------|
| time_stamp_sec | Time stamp for end of window (sec) |
| window_duration_sec | Duration of time window (sec) |
| avg_speed_kmh | Averaged speed of vehicle in the time window (km/h) | 
| grade | Incline of the road as ratio |
| vehicle_mass | Weight of vehicle (kg) |
| battery_capacity_kwh | Total battery capacity (kWh) |
| battery_energy_spend_kwh | Total energy spent from battery in the time window (kWh) |
| motion_energy_drain_kwh | Energy drained for horizontal vehicle motion due to rolling resistance (kWh) |
| gravity_energy_drain_kwh | Energy drained for overcoming gravity on inclines (kWh) |
| heat_energy_drain_kwh | Energy loss due to battery heat dissipation (kWh) |
| aux_energy_drain_kwh | Auxiliary energy consumption by HVAC, HMI, and body control ECUs (kWh) |
| regen_energy_gain_kwh | Energy returned by regenerative braking (kWh) |
| soc_percent | State of charge of the battery (%) |
| distance_km | Distance traveled in the time window (km) |
| remaining_energy_kwh | Remaining battery energy based on current SOC (kWh) |
| remaining_range_km | Estimated remaining range based on current SOC and energy consumption rate (km) |

### ML Dataset (`ev_ml_dataset.csv`)

**Features:**
| Feature | Description |
|---------|-------------|
| avg_speed_kmh | Vehicle speed (km/h) |
| grade | Road incline (ratio) |
| vehicle_mass | Vehicle mass (kg) |
| battery_capacity_kwh | Battery capacity (kWh) |
| soc_percent | State of charge (%) |
| distance_km | Distance in time window (km) |

**Targets:**
| Target | Description |
|--------|-------------|
| remaining_energy_kwh | Remaining battery energy (kWh) |
| remaining_range_km | Remaining driving range (km) |

## Training Data Generation

The generator creates diverse training data by varying:
- Initial SOC: 20% to 100%
- Speed: 20 to 120 km/h
- Grade: -10% to +10%
- Vehicle mass: 1,200 to 2,500 kg
- Battery capacity: 40 to 200 kWh

Two scenario types are generated:
1. **Constant speed** - Fixed speed and grade throughout the trip
2. **Varying speed** - Realistic speed and grade variations over time

## Applications
- Training ML models to predict remaining battery energy and range
- Federated learning of Range Prediction Models
- Validating physical constants for energy efficiency
- Can be modified for other physical processes that involve energy transformations 
