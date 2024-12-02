// webrtc.js
const { RTCPeerConnection, RTCSessionDescription } = require('wrtc');
const winston = require('winston');

class WebRTCHandler {
  constructor(io) {
    this.io = io;
    this.logger = winston.createLogger({
      level: 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      ),
      transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'webrtc.log' }),
      ],
    });
    this.init();
  }

  init() {
    this.io.on('connection', (socket) => {
      this.logger.info('Client connected');

      const peerConnection = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      });

      peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
          socket.emit('ice-candidate', event.candidate);
        }
      };

      peerConnection.ondatachannel = (event) => {
        const dataChannel = event.channel;
        dataChannel.onopen = () => {
          this.logger.info('Data channel opened');
          socket.dataChannel = dataChannel;
        };
        dataChannel.onmessage = (event) => {
          this.logger.info('Received message on data channel');
          // Handle incoming data if needed
        };
      };

      socket.on('offer', async (offer) => {
        try {
          await peerConnection.setRemoteDescription(new RTCSessionDescription(offer));
          const answer = await peerConnection.createAnswer();
          await peerConnection.setLocalDescription(answer);
          socket.emit('answer', peerConnection.localDescription);
          this.logger.info('Sent answer to client');
        } catch (error) {
          this.logger.error('Error handling offer:', error);
        }
      });

      socket.on('ice-candidate', async (candidate) => {
        try {
          await peerConnection.addIceCandidate(candidate);
          this.logger.info('Added ICE candidate from client');
        } catch (error) {
          this.logger.error('Error adding ICE candidate:', error);
        }
      });

      socket.on('disconnect', () => {
        this.logger.info('Client disconnected');
        peerConnection.close();
      });
    });
  }
}

module.exports = WebRTCHandler;
