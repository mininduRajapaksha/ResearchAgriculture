// controllers/logisticsController.js
const GoogleMapService = require('../services/googleMapService');
const WeatherService = require('../services/weatherService');
const VehicleService = require('../services/vehicleService');

/**
 * Helper function: Calculates the distance in km between two coordinates using the Haversine formula.
 */
const calculateDistance = (lat1, lng1, lat2, lng2) => {
  const R = 6371; // Earth radius in km
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180; 
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) *
      Math.cos(lat2 * Math.PI / 180) *
      Math.sin(dLng / 2) *
      Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
};

/**
 * Helper function: Clusters destinations that are within a given threshold (in km).
 * For each destination, it checks if it can be added to an existing group.
 */
const clusterDestinations = (destinations, threshold = 50) => {
  const groups = [];
  destinations.forEach(dest => {
    let added = false;
    for (let group of groups) {
      // If destination is within threshold km of any destination in the group, add it.
      if (
        group.some(existingDest => {
          return calculateDistance(dest.lat, dest.lng, existingDest.lat, existingDest.lng) <= threshold;
        })
      ) {
        group.push(dest);
        added = true;
        break;
      }
    }
    if (!added) {
      groups.push([dest]);
    }
  });
  return groups;
};

class LogisticsController {
  /**
   * Main controller method that processes logistics calculations with grouping based on proximity.
   * Expects a request body in the following format:
   *
   * {
   *   "start": { "lat": 6.745142, "lng": 80.129544 },
   *   "destinations": [
   *      { "lat": ..., "lng": ..., "deliveries": [ { "volume_cft": 2465, "weight_kg": 3000 }, ... ] },
   *      ...
   *   ]
   * }
   */
  static async calculateLogistics(req, res) {
    try {
      const { start, destinations } = req.body;
      if (!start || !destinations || destinations.length === 0) {
        return res.status(400).json({
          error: "Invalid input. Provide a starting point and at least one destination."
        });
      }

      let responseData = {};

      // If only one destination is provided, proceed with an individual route.
      if (destinations.length === 1) {
        const routeData = await GoogleMapService.getIndividualRoute(start, destinations[0]);
        const weatherData = await WeatherService.getWeather(destinations[0].lat, destinations[0].lng);
        const isBadWeather = WeatherService.isBadWeather(weatherData);
        const vehicleRecommendation = await VehicleService.getVehicleRecommendations(
          destinations[0].deliveries,
          routeData.distanceKm
        );

        responseData = {
          type: "individual",
          destination: destinations[0],
          route: routeData,
          weather: weatherData,
          isBadWeather,
          vehicleRecommendation
        };
      } else {
        // Cluster the destinations using a 50 km threshold.
        const groups = clusterDestinations(destinations, 50);
        const groupedRoutes = [];

        // Process each group separately.
        for (let i = 0; i < groups.length; i++) {
          const group = groups[i];
          let routeData;

          // For multiple destinations in a group, calculate a combined route.
          if (group.length > 1) {
            routeData = await GoogleMapService.getRoute(start, group);
          } else {
            // For a single destination group, use the individual route.
            routeData = await GoogleMapService.getIndividualRoute(start, group[0]);
          }

          // Retrieve weather information for each destination in the group.
          let weatherData;
          if (group.length > 1) {
            weatherData = await Promise.all(
              group.map(dest => WeatherService.getWeather(dest.lat, dest.lng))
            );
          } else {
            weatherData = await WeatherService.getWeather(group[0].lat, group[0].lng);
          }

          // Aggregate deliveries from all destinations in the group.
          let aggregatedDeliveries = [];
          group.forEach(dest => {
            if (dest.deliveries && Array.isArray(dest.deliveries)) {
              aggregatedDeliveries = aggregatedDeliveries.concat(dest.deliveries);
            }
          });

          // Use the appropriate distance property from routeData.
          const distance = routeData.distanceKm || routeData.totalDistanceKm;

          // Get vehicle recommendations based on the aggregated deliveries and the route distance.
          const vehicleRecommendation = await VehicleService.getVehicleRecommendations(
            aggregatedDeliveries,
            distance
          );

          groupedRoutes.push({
            groupId: i + 1,
            destinations: group,
            route: routeData,
            weather: weatherData,
            vehicleRecommendation
          });
        }

        responseData = { groupedRoutes };
      }

      return res.json(responseData);
    } catch (error) {
      console.error("Error in calculateLogistics:", error);
      return res.status(500).json({ error: "Internal server error" });
    }
  }
}

module.exports = LogisticsController;
