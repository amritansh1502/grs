// src/components/Results.jsx
import React from 'react';

function Results({ tracks }) {
  if (!tracks || tracks.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2 max-h-96 overflow-y-auto">
      <h4 className="text-lg font-semibold">Your Group Playlist:</h4>
      {tracks.map((track, index) => (
        <div
          key={track.track_id || index}
          className="bg-gray-700 p-3 rounded-lg"
        >
          <div>
            <div className="font-semibold">{track.track_name}</div>
            {/* You can add more data from your CSV to show here if you like */}
          </div>
        </div>
      ))}
    </div>
  );
}

export default Results;