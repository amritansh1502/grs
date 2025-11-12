// src/components/SeedList.jsx
import React from 'react';

function SeedList({ tracks }) {
  if (!tracks || tracks.length === 0) {
    return <div className="text-gray-500 text-center p-4">Add songs to get started!</div>;
  }

  return (
    <div className="space-y-2 max-h-60 overflow-y-auto">
      {tracks.map((track) => (
        <div
          key={track.trackId}
          className="bg-gray-700 p-3 rounded-lg flex justify-between items-center"
        >
          <div>
            <div className="font-semibold">{track.name}</div>
            <div className="text-sm text-gray-400">{track.artist}</div>
          </div>
          <span className="text-green-400">✓ Added</span>
        </div>
      ))}
    </div>
  );
}

export default SeedList;