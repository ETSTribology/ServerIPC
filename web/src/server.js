const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const redis = require('redis');
const zlib = require('zlib');
const BSON = require('bson');
const WebRTCHandler = require('./webrtc');
const winston = require('winston');
const os = require('os');

// Initialize logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'combined.log' }),
  ],
});

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Serve static files from 'public' directory
app.use(express.static('public'));

// Set up Redis client and subscriber
const redisSubscriber = redis.createClient({
  url: 'redis://localhost:6379',
});

const redisPublisher = redisSubscriber.duplicate();

redisSubscriber.on('error', (err) => {
  logger.error('Redis Subscriber Error:', err);
});

redisPublisher.on('error', (err) => {
  logger.error('Redis Publisher Error:', err);
});

// Connect to Redis with retry mechanism
async function connectRedis() {
  while (true) {
    try {
      await redisSubscriber.connect();
      await redisPublisher.connect();
      logger.info('Connected to Redis');
      await redisSubscriber.subscribe('simulation_updates', async (message) => {
        try {
          const meshDataCompressed = Buffer.from(message, 'base64');
          const meshDataBson = zlib.unzipSync(meshDataCompressed);
          const meshData = BSON.deserialize(meshDataBson);

          // Convert Binary objects to base64 strings for JSON serialization
          const serializedMeshData = {
            ...meshData,
            BX: meshData.BX.buffer.toString('base64'),
            faces: meshData.faces.buffer.toString('base64')
          };

          io.emit('meshData', JSON.stringify(serializedMeshData));
        } catch (error) {
          logger.error('Error processing mesh data:', error);
        }
      });
      break;
    } catch (error) {
      logger.error('Redis connection failed, retrying in 5 seconds...', error);
      await new Promise(res => setTimeout(res, 5000));
    }
  }
}

connectRedis();

// Initialize WebRTC Handler
new WebRTCHandler(io);

// Add a new route to handle the start command
app.post('/start-simulation', async (req, res) => {
  try {
    await redisPublisher.publish('simulation_commands', 'start');
    logger.info('Start command sent to simulation');
    res.status(200).json({ message: 'Start command sent successfully' });
  } catch (error) {
    logger.error('Error sending start command:', error);
    res.status(500).json({ error: 'Failed to send start command' });
  }
});

// Add a new route to handle the stop command
app.post('/stop-simulation', async (req, res) => {
  try {
    await redisPublisher.publish('simulation_commands', 'stop');
    logger.info('Stop command sent to simulation');
    res.status(200).json({ message: 'Stop command sent successfully' });
  } catch (error) {
    logger.error('Error sending stop command:', error);
    res.status(500).json({ error: 'Failed to send stop command' });
  }
});

// Function to get local IP address
function getLocalIpAddress() {
  const interfaces = os.networkInterfaces();
  for (const devName in interfaces) {
    const iface = interfaces[devName];
    for (let i = 0; i < iface.length; i++) {
      const alias = iface[i];
      if (alias.family === 'IPv4' && alias.address !== '127.0.0.1' && !alias.internal) {
        return alias.address;
      }
    }
  }
  return '0.0.0.0';
}

// Start the server
const PORT = process.env.PORT || 3001;
const HOST = '0.0.0.0';
const localIpAddress = getLocalIpAddress();

let serverInstance = server.listen(PORT, HOST, () => {
  logger.info(`Server is running on port ${PORT}`);
  logger.info(`Local IP address: http://${localIpAddress}:${PORT}`);
  logger.info(`You can also access the server at http://localhost:${PORT}`);
});

// Handle process events for graceful shutdown
let isShuttingDown = false;

async function gracefulShutdown() {
  if (isShuttingDown) {
    logger.info('Shutdown already in progress');
    return;
  }
  isShuttingDown = true;

  logger.info('Initiating graceful shutdown...');

  // Close Socket.IO connections
  io.close(() => {
    logger.info('Socket.IO connections closed');

    // Close the server
    serverInstance.close(async (err) => {
      if (err) {
        logger.error('Error closing the server:', err);
      } else {
        logger.info('Server closed successfully');
      }

      // Disconnect Redis clients
      try {
        await redisSubscriber.quit();
        await redisPublisher.quit();
        logger.info('Redis connections closed');
      } catch (err) {
        logger.error('Error closing Redis connections:', err);
      }

      // Exit the process
      process.exit(err ? 1 : 0);
    });
  });

  // If server hasn't closed in 10 seconds, force shutdown
  setTimeout(() => {
    logger.error('Could not close connections in time, forcefully shutting down');
    process.exit(1);
  }, 10000);
}

process.on('SIGINT', gracefulShutdown);
process.on('SIGTERM', gracefulShutdown);

// Catch unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
  // Application specific logging, throwing an error, or other logic here
});
