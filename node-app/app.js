// index.js
const express = require('express');
const bodyParser = require('body-parser');
const logisticsRoutes = require('./routes/logisticsRoutes');

const app = express();
app.use(bodyParser.json());

// Mount the logistics routes
app.use('/api/logistics', logisticsRoutes);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
