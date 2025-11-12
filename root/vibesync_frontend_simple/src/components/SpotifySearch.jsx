// src/components/SpotifySearch.jsx
import React, { useState, useEffect } from 'react';
import { searchTracks } from '../api';

function SpotifySearch({ onTrackAdded }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  // This is a "debouncer". It waits 500ms after you stop typing to search.
  useEffect(() => {
    console.log('Query changed:', query);
    if (query.length < 3) {
      setResults([]);
      return;
    }

    setIsSearching(true);
    const delayDebounceFn = setTimeout(() => {
      const performSearch = async () => {
        console.log('Performing search for:', query);
        try {
          const response = await searchTracks(query);
          console.log('Search results:', response);
          setResults(response.data);
        } catch (error) {
          console.error('Search failed:', error);
        }
        setIsSearching(false);
      };
      performSearch();
    }, 500);

    return () => clearTimeout(delayDebounceFn); // Cleanup on unmount
  }, [query]);

  const handleAddClick = (track) => {
    // Pass the track data up to RoomPage.jsx
    onTrackAdded({
      trackId: track.trackId,
      name: track.name,
      artist: track.artist,
    });
    // Clear search
    setQuery('');
    setResults([]);
  };

  return (
    <div className="w-full">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search for a song or artist..."
        className="w-full bg-gray-700 text-white placeholder-gray-500 p-3 rounded-lg border border-gray-600 focus:outline-none focus:ring-2 focus:ring-green-500"
      />
      {isSearching && <div className="text-gray-400 p-2">Searching...</div>}
      
      {results.length > 0 && (
        <ul className="bg-gray-700 rounded-lg mt-2 max-h-60 overflow-y-auto">
          {results.map((track) => (
            <li
              key={track.trackId}
              className="flex items-center justify-between p-3 border-b border-gray-600 last:border-b-0"
            >
              <div className="flex items-center space-x-3">
                <img src={track.albumArt} alt={track.name} className="w-10 h-10 rounded" />
                <div>
                  <div className="font-semibold">{track.name}</div>
                  <div className="text-sm text-gray-400">{track.artist}</div>
                </div>
              </div>
              <button
                onClick={() => handleAddClick(track)}
                className="bg-green-600 hover:bg-green-700 text-white font-bold py-1 px-3 rounded-full text-sm"
              >
                Add
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default SpotifySearch;