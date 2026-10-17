import React, { useState, useRef } from 'react'
import { Eye, EyeOff } from 'lucide-react'

export default function BBoxCanvas({ imageUrl, rawDetections, imageMetadata }) {
  const [showBoxes, setShowBoxes] = useState(true)
  const [hoveredIndex, setHoveredIndex] = useState(null)
  const [tooltip, setTooltip] = useState({ visible: false, x: 0, y: 0, text: '', conf: 0 })
  const containerRef = useRef(null)

  const origW = imageMetadata?.width || 630
  const origH = imageMetadata?.height || 740

  const handlePolygonMouseEnter = (e, det, index) => {
    setHoveredIndex(index)
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect()
      setTooltip({
        visible: true,
        x: e.clientX - rect.left + 10,
        y: e.clientY - rect.top - 30,
        text: det.text,
        conf: Math.round(det.confidence * 100),
      })
    }
  }

  const handlePolygonMouseLeave = () => {
    setHoveredIndex(null)
    setTooltip((prev) => ({ ...prev, visible: false }))
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
          {rawDetections.length > 0
            ? `Detected ${rawDetections.length} text bounding boxes`
            : 'Image Preview'}
        </span>
        {rawDetections.length > 0 && (
          <button
            className="btn-secondary"
            style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem' }}
            onClick={() => setShowBoxes(!showBoxes)}
          >
            {showBoxes ? <EyeOff size={14} /> : <Eye size={14} />}
            <span>{showBoxes ? 'Hide Bounding Boxes' : 'Show Bounding Boxes'}</span>
          </button>
        )}
      </div>

      <div className="canvas-wrapper">
        {imageUrl ? (
          <div
            ref={containerRef}
            style={{
              position: 'relative',
              display: 'inline-block',
              maxWidth: '100%',
              lineHeight: 0,
            }}
          >
            <img
              src={imageUrl}
              alt="Uploaded Document"
              style={{
                display: 'block',
                maxWidth: '100%',
                maxHeight: '480px',
                height: 'auto',
                borderRadius: '8px',
                objectFit: 'contain',
              }}
            />

            {showBoxes && rawDetections.length > 0 && (
              <svg
                viewBox={`0 0 ${origW} ${origH}`}
                preserveAspectRatio="none"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'all',
                }}
              >
                {rawDetections.map((det, idx) => {
                  if (!det.box || det.box.length < 4) return null
                  const pointsStr = det.box.map((pt) => `${pt[0]},${pt[1]}`).join(' ')
                  const isHovered = hoveredIndex === idx

                  return (
                    <polygon
                      key={idx}
                      points={pointsStr}
                      className={`bbox-polygon ${isHovered ? 'highlighted' : ''}`}
                      onMouseEnter={(e) => handlePolygonMouseEnter(e, det, idx)}
                      onMouseLeave={handlePolygonMouseLeave}
                    />
                  )
                })}
              </svg>
            )}

            {tooltip.visible && (
              <div
                className="bbox-tooltip"
                style={{
                  left: `${tooltip.x}px`,
                  top: `${tooltip.y}px`,
                  lineHeight: '1.4',
                }}
              >
                <strong>{tooltip.text}</strong>
                <div style={{ color: '#818cf8', fontSize: '0.7rem' }}>
                  Confidence: {tooltip.conf}%
                </div>
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: '#64748b', fontSize: '0.9rem' }}>
            No image loaded. Upload or select a preset above.
          </div>
        )}
      </div>
    </div>
  )
}
