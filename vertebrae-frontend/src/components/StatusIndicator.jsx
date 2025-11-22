import { useEffect, useState } from 'react';
import api from '../services/api';
import './StatusIndicator.css';

export default function StatusIndicator() {
  const [status, setStatus] = useState({
    connected: false,
    checking: true,
    message: 'Checking API status...',
  });

  useEffect(() => {
    checkHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const health = await api.checkHealth();
      setStatus({
        connected: health.status === 'healthy',
        checking: false,
        message: health.status === 'healthy' ? 'API Connected' : 'API Unhealthy',
      });
    } catch (error) {
      setStatus({
        connected: false,
        checking: false,
        message: 'API Disconnected',
      });
    }
  };

  return (
    <div className="status-card">
      <div className="status-indicator">
        <span
          className={`status-dot ${
            status.checking ? 'checking' : status.connected ? 'online' : 'offline'
          }`}
        ></span>
        <span>{status.message}</span>
      </div>
    </div>
  );
}
