import React from 'react'
import { Activity } from 'lucide-react'

export default function LatencyGauge({ latency }) {
  if (!latency) return null

  const total = latency.total_ms || 0
  const isOptimal = total < 1000

  return (
    <div className="latency-container">
      <div className="latency-header">
        <div className="latency-title">
          <Activity size={15} color="#38bdf8" />
          <span>CPU EXECUTION TELEMETRY</span>
        </div>
        <div className="total-badge">
          TOTAL: {total.toFixed(1)} ms {isOptimal && '• TARGET PASS (< 1s)'}
        </div>
      </div>

      <div className="latency-grid">
        <div className="latency-metric">
          <div className="metric-label">PREPROCESS</div>
          <div className="metric-value">{(latency.preprocessing_ms || 0).toFixed(1)} ms</div>
        </div>

        <div className="latency-metric">
          <div className="metric-label">DBNET DET</div>
          <div className="metric-value" style={{ color: '#38bdf8' }}>
            {(latency.detection_ms || 0).toFixed(1)} ms
          </div>
        </div>

        <div className="latency-metric">
          <div className="metric-label">SVTR REC</div>
          <div className="metric-value" style={{ color: '#0ea5e9' }}>
            {(latency.recognition_ms || 0).toFixed(1)} ms
          </div>
        </div>

        <div className="latency-metric">
          <div className="metric-label">KV PARSE</div>
          <div className="metric-value" style={{ color: '#10b981' }}>
            {(latency.parsing_ms || 0).toFixed(1)} ms
          </div>
        </div>
      </div>
    </div>
  )
}
