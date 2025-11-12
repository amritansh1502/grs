// src/api.js
import axios from 'axios';

// Your Node.js API (from Phase 3)
const API_URL = '/api';

const api = axios.create({
  baseURL: API_URL,
});

// Endpoints
export const searchTracks = (query) => api.get('/search', { params: { q: query } });

// This now sends both mood and tracks
export const generateRecommendations = (mood_vote, seedTracks) => {
  return api.post('/generate-playlist', { 
    mood_vote: mood_vote,
    seedTracks: seedTracks 
  });
};

export default api;