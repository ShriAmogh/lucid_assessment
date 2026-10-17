import React from 'react'
import { ScanText, Cpu, CheckCircle2 } from 'lucide-react'

export default function Header({ statusInfo }) {
  return (
    <header className="navbar">
      <div className="logo-group">
        <div className="logo-icon-box">
          <ScanText size={20} color="#38bdf8" />
        </div>
        <div>
          <div className="brand-title">OCR Engine</div>
          <div className="brand-meta">
            <span className="brand-tag">ONNX CPU</span>
            <span className="brand-tag">OFFLINE PIPELINE</span>
            <span className="brand-tag">INTRA-OP 4T</span>
          </div>
        </div>
      </div>

      <div className="system-status">
        <div className="status-pill">
          <span className="pulse-dot"></span>
          <span>Engine Online</span>
        </div>

        <div className="threads-badge">
          <Cpu size={14} color="#38bdf8" />
          <span>{statusInfo?.cpu_threads || 4} Physical Threads</span>
        </div>
      </div>
    </header>
  )
}
