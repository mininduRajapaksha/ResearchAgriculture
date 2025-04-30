import pandas as pd
import random

vehicles = [
    {"name": "TATA ACE", "max_weight": 850, "max_volume": 7 * 4.8 * 4.8},
    {"name": "ASHOK LEYLAND DOST", "max_weight": 1000, "max_volume": 7 * 4.8 * 4.8},
    {"name": "MAHINDRA BOLERO PICK UP", "max_weight": 1500, "max_volume": 8 * 5 * 4.8},
    {"name": "ASHOK LEYLAND BADA DOST", "max_weight": 2000, "max_volume": 9.5 * 5.5 * 5},
    {"name": "TATA 407", "max_weight": 2500, "max_volume": 9 * 5.5 * 5},
    {"name": "EICHER 14 FEET", "max_weight": 4000, "max_volume": 14 * 6 * 6.5},
    {"name": "EICHER 17 FEET", "max_weight": 5000, "max_volume": 17 * 6 * 7},
    {"name": "EICHER 19 FEET", "max_weight": 9000, "max_volume": 19 * 7 * 7},
    {"name": "TATA 22 FEET", "max_weight": 10000, "max_volume": 22 * 7.5 * 7},
    {"name": "TATA TRUCK (6 TYRE)", "max_weight": 9000, "max_volume": 17.5 * 7 * 7},
    {"name": "CONTAINER 20 FT", "max_weight": 6500, "max_volume": 20 * 7.5 * 7.5},
    {"name": "CONTAINER 22 FT", "max_weight": 10000, "max_volume": 22 * 8 * 8},
    {"name": "CONTAINER 32 FT SXL", "max_weight": 7000, "max_volume": 32 * 8 * 8},
    {"name": "CONTAINER 32 FT MXL", "max_weight": 18000, "max_volume": 32 * 8 * 9},
    {"name": "20 FEET OPEN ALL SIDE (ODC)", "max_weight": 7000, "max_volume": 20 * 8 * 8},
    {"name": "28-32 FEET OPEN JCB TRUCK ODC", "max_weight": 10000, "max_volume": 32 * 8 * 8}
]

# Select smallest vehicle satisfying constraints
def select_vehicle(volume, weight):
    suitable = sorted(
        [v for v in vehicles if v["max_volume"] >= volume and v["max_weight"] >= weight],
        key=lambda x: (x["max_volume"], x["max_weight"])
    )
    return suitable[0]["name"] if suitable else "TATA 22 FEET"

def generate_logistics_data(num_samples=500):
    data = []
    for _ in range(num_samples):
        volume = random.randint(50, 2300)  # Realistic volumes for your dataset
        weight = random.randint(100, 17000)  # Realistic weights for your vehicles
        distance = random.randint(10, 500)
        vehicle = select_vehicle(volume, weight)

        data.append({
            "volume_cft": volume,
            "weight_kg": weight,
            "distance_km": distance,
            "optimal_vehicle": vehicle
        })

    df = pd.DataFrame(data)
    df.to_csv("logistics_data.csv", index=False)
    print("✅ Dataset generated successfully: logistics_data.csv")

if __name__ == "__main__":
    generate_logistics_data(500)
