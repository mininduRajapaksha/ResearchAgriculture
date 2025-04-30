from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
from data.vehicles import vehicles

app = Flask(__name__)
CORS(app)  
model = joblib.load('model/vehicle_selector_model.pkl')

vehicle_costs_per_km = {v['name']: v['cost_per_km'] for v in vehicles}

def find_single_vehicle(volume, weight):
    suitable = sorted(
        [v for v in vehicles if v["max_volume"] >= volume and v["max_weight"] >= weight],
        key=lambda x: (x["cost_per_km"], x["max_volume"], x["max_weight"])
    )
    return suitable[0] if suitable else None

def allocate_multiple_vehicles(volume, weight):
    allocation = []
    remaining_volume = volume
    remaining_weight = weight

    sorted_vehicles = sorted(vehicles, key=lambda x: (x["max_volume"], x["max_weight"]), reverse=True)

    for veh in sorted_vehicles:
        while remaining_volume > 0 or remaining_weight > 0:
            if veh["max_volume"] <= remaining_volume or veh["max_weight"] <= remaining_weight:
                allocation.append(veh)
                remaining_volume -= min(veh["max_volume"], remaining_volume)
                remaining_weight -= min(veh["max_weight"], remaining_weight)
            else:
                break

    return allocation

# Function to determine the optimal vehicle strategy
def optimal_vehicle_strategy(volume, weight, distance):
    single_vehicle = find_single_vehicle(volume, weight)

    cost_single = single_vehicle["cost_per_km"] * distance if single_vehicle else float('inf')

    multiple_vehicles = allocate_multiple_vehicles(volume, weight)

    cost_multi = sum([veh["cost_per_km"] for veh in multiple_vehicles]) * distance if multiple_vehicles else float('inf')

    if cost_single <= cost_multi:
        return {
            "strategy": "Single Vehicle",
            "vehicles": [single_vehicle["name"]],
            "estimated_cost_LKR": cost_single
        }
    else:
        return {
            "strategy": "Multiple Vehicles",
            "vehicles": [v for v in [veh["name"] for veh in multiple_vehicles]],
            "estimated_cost_LKR": cost_multi
        }

@app.route('/predict', methods=['POST'])
def recommend_vehicle():
    data = request.json
    deliveries = data['deliveries']
    distance = data['distance_km']

    total_volume = sum(d['volume_cft'] for d in deliveries)
    total_weight = sum(d['weight_kg'] for d in deliveries)

    optimal_strategy = optimal_vehicle_strategy(total_volume, total_weight, distance)

    return jsonify({
        "strategy": optimal_strategy['strategy'],
        "vehicles": optimal_strategy['vehicles'],
        "total_estimated_cost_LKR": optimal_strategy['estimated_cost_LKR'],
        "total_distance_km": distance,
        "total_volume_cft": total_volume,
        "total_weight_kg": total_weight
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
