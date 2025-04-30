// services/vehicleService.js
const axios = require('axios');
const { MODEL_API_ENDPOINT } = require('../constants');

class VehicleService {
  /**
   * Calls the AI model API to get vehicle recommendations.
   */
  static async getVehicleRecommendations(deliveries, distanceKm) {
    try {
      const response = await axios.post(MODEL_API_ENDPOINT, {
        deliveries,
        distance_km: distanceKm
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  }
}

module.exports = VehicleService;
