// services/googleMapService.js
const axios = require('axios');
const { OPENROUTESERVICE_API_KEY } = require('../constants');

class GoogleMapService {
  /**
   * Calculates the route-based driving distance (in kilometers) between two coordinates
   * using the OpenRouteService Matrix API.
   */
  static async calculateRouteDistance(start, destination) {
    try {
      const url = "https://api.openrouteservice.org/v2/matrix/driving-car";
      const payload = {
        // OpenRouteService expects coordinates in [longitude, latitude] order.
        locations: [
          [start.lng, start.lat],
          [destination.lng, destination.lat]
        ],
        metrics: ["distance"],
        units: "km"
      };

      const headers = {
        "Authorization": OPENROUTESERVICE_API_KEY,
        "Content-Type": "application/json"
      };

      const response = await axios.post(url, payload, { headers });

      if (response.data && response.data.distances && response.data.distances[0]) {
        // Distance from origin (index 0) to destination (index 1)
        const distanceKm = response.data.distances[0][1];
        return distanceKm;
      } else {
        throw new Error("No distance data found in response.");
      }
    } catch (error) {
      console.error(
        "Error in calculateRouteDistance:",
        error.response ? error.response.data : error.message
      );
      throw error;
    }
  }

  /**
   * For multiple destinations, this method calculates the overall route distance
   * using the OpenRouteService Directions API.
   * It builds the coordinates list from the start and destinations (in order),
   * then calculates the route distance (in km) and constructs a Google Maps route link.
   */
  static async getRoute(start, destinations) {
    try {
      // Build an array of coordinates for the route.
      // Note: OpenRouteService expects [lng, lat] order.
      const coordinates = [
        [start.lng, start.lat],
        ...destinations.map(dest => [dest.lng, dest.lat])
      ];

      const url = "https://api.openrouteservice.org/v2/directions/driving-car";
      const headers = {
        "Authorization": OPENROUTESERVICE_API_KEY,
        "Content-Type": "application/json"
      };

      const payload = {
        coordinates: coordinates
        // Optionally, you can add additional parameters (e.g., instructions: false)
      };

      const response = await axios.post(url, payload, { headers });

      if (response.data && response.data.routes && response.data.routes[0]) {
        const routeSummary = response.data.routes[0].summary;
        // OpenRouteService returns the distance in meters; convert to kilometers.
        const totalDistanceKm = routeSummary.distance / 1000;

        // Build a Google Maps link.
        // For Google Maps, origin is the start, destination is the last coordinate,
        // and intermediate points are provided as waypoints.
        const finalDestination = destinations[destinations.length - 1];
        // If there are more than one destination, the waypoints are all destinations except the last.
        const waypoints = destinations.length > 1
          ? destinations.slice(0, -1).map(dest => `${dest.lat},${dest.lng}`).join('|')
          : '';
        const routeLink = `https://www.google.com/maps/dir/?api=1&origin=${start.lat},${start.lng}&destination=${finalDestination.lat},${finalDestination.lng}${waypoints ? `&waypoints=${waypoints}` : ''}&travelmode=driving`;

        return { totalDistanceKm, routeLink };
      } else {
        throw new Error("No route found.");
      }
    } catch (error) {
      console.error(
        "Error in getRoute:",
        error.response ? error.response.data : error.message
      );
      throw error;
    }
  }

  /**
   * Returns route info for an individual destination.
   * It calculates the route-based distance using the OpenRouteService Matrix API
   * and constructs a Google Maps route link.
   */
  static async getIndividualRoute(start, destination) {
    try {
      const distanceKm = await GoogleMapService.calculateRouteDistance(start, destination);
      const routeLink = `https://www.google.com/maps/dir/?api=1&origin=${start.lat},${start.lng}&destination=${destination.lat},${destination.lng}&travelmode=driving`;
      return { distanceKm, routeLink };
    } catch (error) {
      throw error;
    }
  }
}

module.exports = GoogleMapService;
