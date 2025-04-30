// routes/logisticsRoutes.js
const express = require('express');
const LogisticsController = require('../controllers/logisticsController');

const router = express.Router();

// POST endpoint to calculate logistics.
router.post('/calculate', LogisticsController.calculateLogistics);

module.exports = router;
