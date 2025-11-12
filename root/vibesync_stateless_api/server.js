const express = require('express');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config();

// --- 1. APP SETUP ---
const app = express();
const PORT = process.env.PORT || 8080;

app.use(cors());
app.use(express.json());

// --- 2. SPOTIFY API HELPERS (FINAL CORRECTED VERSION) ---

const SPOTIFY_TOKEN_URL = 'https://accounts.spotify.com/api/token';
const SPOTIFY_API_BASE_URL = 'https://api.spotify.com/v1';
const ML_API_URL = process.env.PYTHON_ML_API_URL;

// These must match the order in your Python API
const FEATURE_COLS = ["danceability", "energy", "valence", "tempo"];

let spotifyToken = null;

const spotifyApi = axios.create({
    baseURL: SPOTIFY_API_BASE_URL,
});

const getSpotifyToken = async () => {
    try {
        const authString = Buffer.from(`${process.env.SPOTIFY_CLIENT_ID}:${process.env.SPOTIFY_CLIENT_SECRET}`).toString('base64');
        const response = await axios.post(SPOTIFY_TOKEN_URL, 'grant_type=client_credentials', {
            headers: {
                'Authorization': `Basic ${authString}`,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        });
        spotifyToken = response.data.access_token;
        const expiresIn = response.data.expires_in || 3600;
        setTimeout(() => { spotifyToken = null; }, (expiresIn * 1000) - 300000); // 5 min buffer
        console.log('New Spotify Token obtained.');
    } catch (error) {
        console.error('Error getting Spotify token:', error.response ? error.response.data : error.message);
    }
};

spotifyApi.interceptors.request.use(async (config) => {
    if (!spotifyToken) {
        console.log("Token is null or expired, fetching new one...");
        await getSpotifyToken();
    }
    config.headers.Authorization = `Bearer ${spotifyToken}`;
    return config;
}, (error) => Promise.reject(error));

// --- 3. API ENDPOINTS ---

// Endpoint 1: Search Spotify (Unchanged)
app.get('/search', async (req, res) => {
    const { q } = req.query;
    if (!q) {
        return res.status(400).json({ error: 'Query parameter "q" is required' });
    }
    try {
        const response = await spotifyApi.get('/search', {
            params: { q, type: 'track', limit: 10 }
        });
        const tracks = response.data.tracks.items.map(item => ({
            trackId: item.id,
            name: item.name,
            artist: item.artists.map(a => a.name).join(', '),
            albumArt: item.album.images[2]?.url
        }));
        res.json(tracks);
    } catch (error) {
        console.error("Error in /search:", error.response ? error.response.data : error.message);
        res.status(500).json({ error: 'Error searching Spotify' });
    }
});

// Endpoint 2: THE "MAGIC" ENDPOINT (Updated for new ML logic)
app.post('/generate-playlist', async (req, res) => {
    // 1. Get mood and tracks from React
    const { seedTracks, mood_vote } = req.body; 
    
    if (!seedTracks || seedTracks.length === 0) {
        return res.status(400).json({ error: 'seedTracks array is empty.' });
    }
    if (!mood_vote) {
        return res.status(400).json({ error: 'mood_vote is missing.' });
    }
    
    try {
        // 2. Get audio features for all seed tracks in parallel
        const featurePromises = seedTracks.map(async (track) => {
            const featuresResponse = await spotifyApi.get(`/audio-features/${track.trackId}`);
            return featuresResponse.data;
        });
        const allFeatures = await Promise.all(featurePromises);

        // 3. Calculate the "group preference vector" [cite: 35]
        // This is the average of all feature columns.
        const avg_vector = [0, 0, 0, 0];
        allFeatures.forEach(features => {
            avg_vector[0] += features.danceability;
            avg_vector[1] += features.energy;
            avg_vector[2] += features.valence;
            avg_vector[3] += features.tempo;
        });
        const numTracks = allFeatures.length;
        const average_feature_vector = avg_vector.map(val => val / numTracks);

        // 4. Call our Python ML API with the mood and the vector
        console.log(`Recommending for: Mood=${mood_vote}, Vector=${average_feature_vector}`);
        const recommendationsResponse = await axios.post(`${ML_API_URL}/recommend`, {
            mood_vote: mood_vote,
            average_feature_vector: average_feature_vector
        });

        // 5. Return the final list to the React frontend!
        res.json(recommendationsResponse.data);

    } catch (error) {
        console.error("Error in /generate-playlist:", error.response ? error.response.data : error.message);
        res.status(500).json({ error: error.message });
    }
});

// --- 4. START SERVER ---
app.listen(PORT, async () => {
    await getSpotifyToken(); // Get initial Spotify token
    console.log(`Stateless Node.js API listening on http://127.0.0.1:${PORT}`);
});