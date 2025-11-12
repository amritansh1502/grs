// src/App.jsx
import React, { useState } from 'react';
import { generateRecommendations } from './api';

// Import the components (no changes needed to these files)
import SpotifySearch from './components/SpotifySearch';
import SeedList from './components/SeedList';
import Results from './components/Results';

// Moods from your report [cite: 28, 65]
const MOODS = ["Happy", "Sad", "Energetic", "Calm"];

function App() {
  // All our app "state" lives here
  const [seedTracks, setSeedTracks] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // NEW: State for the mood vote 
  const [moodVote, setMoodVote] = useState(MOODS[0]); // Default to "Happy"

  // --- Child Component Callbacks ---

  const handleTrackAdded = (trackData) => {
    // Add the new track to our local state array
    setSeedTracks(prevTracks => [...prevTracks, trackData]);
  };

  const handleGenerate = async () => {
    setIsLoading(true);
    setError(null);
    setRecommendations([]);
    
    try {
      // Send both the mood and the tracks to the backend [cite: 32, 34]
      const response = await generateRecommendations(moodVote, seedTracks);
      setRecommendations(response.data.recommendations);
    } catch (err) {
      console.error('Error generating recommendations:', err);
      setError('Could not generate recommendations. Try again.');
    }
    setIsLoading(false);
  };

  // --- Render Logic ---

  return (
    <div className="min-h-screen container mx-auto p-4 md:p-8">
      <header className="text-center mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-blue-500">
          VibeSync
        </h1>
        <p className="text-lg text-gray-400">Group Vibe Music Recommender [cite: 1]</p>
      </header>

      <main className="max-w-4xl mx-auto space-y-8">
        
        {/* --- STEP 1: VOTE FOR A MOOD --- */}
        <div className="bg-gray-800 rounded-lg shadow-xl p-6 space-y-4">
          <h3 className="text-2xl font-semibold border-b border-gray-700 pb-2">Step 1: Vote for a Vibe </h3>
          <div className="flex flex-wrap gap-3">
            {MOODS.map((mood) => (
              <button
                key={mood}
                onClick={() => setMoodVote(mood)}
                className={`font-medium py-2 px-5 rounded-full transition-all duration-200
                  ${moodVote === mood 
                    ? 'bg-green-500 text-white shadow-lg' 
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }
                `}
              >
                {mood}
              </button>
            ))}
          </div>
        </div>

        {/* --- STEP 2: ADD SEED SONGS --- */}
        <div className="bg-gray-800 rounded-lg shadow-xl p-6 space-y-4">
          <h3 className="text-2xl font-semibold border-b border-gray-700 pb-2">Step 2: Add Seed Songs (Optional) [cite: 30]</h3>
          <SpotifySearch onTrackAdded={handleTrackAdded} />
          <SeedList tracks={seedTracks} />
        </div>
        
        {/* --- STEP 3: GENERATE --- */}
        <div className="bg-gray-800 rounded-lg shadow-xl p-6 space-y-4">
          <h3 className="text-2xl font-semibold border-b border-gray-700 pb-2">Step 3: Generate Playlist</h3>
          <button
            onClick={handleGenerate}
            disabled={isLoading || seedTracks.length === 0}
            className="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-4 rounded-lg text-lg transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Generating...' : `Generate for "${moodVote}" (${seedTracks.length} seeds)`}
          </button>
        </div>
        
        {/* --- STEP 4: RESULTS --- */}
        {error && <div className="text-center text-red-400 text-xl">{error}</div>}
        {isLoading && <div className="text-center text-lg">Calculating group vector...</div>}
        
        {recommendations.length > 0 && (
          <div className="bg-gray-800 rounded-lg shadow-xl p-6">
             <h3 className="text-2xl font-semibold border-b border-gray-700 pb-2 mb-4">Your Group Playlist</h3>
            <Results tracks={recommendations} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;