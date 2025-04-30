// services/weatherService.js
const axios = require('axios');
const { OPEN_WEATHER_API_KEY } = require('../constants');

class WeatherService {
  /**
   * Fetch current weather for the provided coordinates.
   */
  static async getWeather(lat, lng) {
    try {
      const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lng}&appid=${OPEN_WEATHER_API_KEY}&units=metric`;
      const response = await axios.get(url);
      return response.data;
    } catch (error) {
      throw error;
    }
  }

  /**
   * Simple check to see if the weather condition is considered “bad.”
   * You can expand the logic based on your criteria.
   */
  static isBadWeather(weatherData) {
    const badConditions = ['Rain', 'Thunderstorm', 'Snow'];
    return weatherData.weather.some(condition =>
      badConditions.includes(condition.main)
    );
  }
}

module.exports = WeatherService;
