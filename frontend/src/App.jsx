import React, { useState, useEffect } from 'react'
import { Upload, Play, AlertCircle, Scan, FileCode, CheckCircle2 } from 'lucide-react'

import Header from './components/Header'
import SamplePicker, { SAMPLES } from './components/SamplePicker'
import BBoxCanvas from './components/BBoxCanvas'
import KeyValueCards from './components/KeyValueCards'
import LatencyGauge from './components/LatencyGauge'
import JsonViewer from './components/JsonViewer'

const API_BASE = '/api'

export default function App() {
  const [statusInfo, setStatusInfo] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [languageHint, setLanguageHint] = useState('auto')
  const [activeSampleId, setActiveSampleId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [isDragOver, setIsDragOver] = useState(false)

  // Fetch initial health status and load sample on startup (supports ?sample=...)
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((res) => res.json())
      .then((data) => setStatusInfo(data))
      .catch((err) => console.error('Health check error:', err))

    const params = new URLSearchParams(window.location.search)
    const targetId = params.get('sample') || 'roman_id_1.png'
    const found = SAMPLES.find((s) => s.id === targetId) || SAMPLES[0]

    loadSample(found)
  }, [])

  // Execute extraction API call
  const runExtraction = async (fileBlob, filename, hint) => {
    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', fileBlob, filename)
      formData.append('language_hint', hint)

      const response = await fetch(`${API_BASE}/extract`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Server returned ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      console.error('Extraction failed:', err)
      setError(err.message || 'Failed to execute OCR inference pipeline.')
    } finally {
      setLoading(false)
    }
  }

  // Load a preloaded sample document
  const loadSample = async (sample) => {
    setActiveSampleId(sample.id)
    setLanguageHint(sample.hint)
    const imgUrl = `${API_BASE}/samples/${sample.id}`
    setPreviewUrl(imgUrl)
    setSelectedFile(null)

    try {
      setLoading(true)
      const res = await fetch(imgUrl)
      const blob = await res.blob()
      await runExtraction(blob, sample.id, sample.hint)
    } catch (err) {
      console.error('Error loading sample:', err)
      setError(`Failed to load preset sample: ${sample.label}`)
      setLoading(false)
    }
  }

  // Handle user uploaded file
  const handleFileUpload = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please upload a valid image file (PNG, JPG, WebP).')
      return
    }

    setActiveSampleId(null)
    setSelectedFile(file)
    const localUrl = URL.createObjectURL(file)
    setPreviewUrl(localUrl)
    runExtraction(file, file.name, languageHint)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0])
    }
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header statusInfo={statusInfo} />

      <main className="main-container">
        {/* Preset Selector Toolbar */}
        <SamplePicker
          activeSampleId={activeSampleId}
          onSelectSample={loadSample}
          loading={loading}
        />

        {/* Error Alert */}
        {error && (
          <div
            style={{
              padding: '0.75rem 1rem',
              background: 'rgba(244, 63, 94, 0.1)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              borderRadius: 'var(--radius-sm)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.6rem',
              color: '#fda4af',
              fontSize: '0.85rem',
            }}
          >
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        {/* Two-column Engineering Dashboard */}
        <div className="dashboard-grid">
          {/* Left Column: Image Viewer, Drag-and-Drop, Controls */}
          <div className="glass-panel viewer-card">
            <div className="card-header-bar">
              <div className="panel-title">
                <Scan size={16} color="#38bdf8" />
                <span>DOCUMENT INSPECTOR</span>
              </div>

              {result?.image_metadata && (
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  {result.image_metadata.width} × {result.image_metadata.height} PX
                </div>
              )}
            </div>

            {/* Compact Drag & Drop Upload Strip */}
            <div
              className={`compact-upload-bar ${isDragOver ? 'dragover' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-input').click()}
            >
              <input
                id="file-input"
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
              />
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Upload size={14} color="#38bdf8" />
                <span className="upload-text">
                  Upload custom document or <span>browse local files</span>
                </span>
              </div>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                PNG / JPG / RGBA
              </span>
            </div>

            {/* Interactive Bounding Box Canvas */}
            <BBoxCanvas
              imageUrl={previewUrl}
              rawDetections={result?.raw_detections || []}
              imageMetadata={result?.image_metadata || {}}
            />

            {/* Language & Trigger Controls */}
            <div className="controls-bar">
              <select
                className="select-input"
                value={languageHint}
                onChange={(e) => setLanguageHint(e.target.value)}
                disabled={loading}
              >
                <option value="auto">Language Routing: Auto-Detect</option>
                <option value="romanian">Romanian (Latin SVTR)</option>
                <option value="german">German (Latin SVTR)</option>
                <option value="greek">Greek (Greek SVTR)</option>
                <option value="english">English / UK (Latin SVTR)</option>
              </select>

              <button
                className="btn-primary"
                disabled={loading || !previewUrl}
                onClick={() => {
                  if (activeSampleId) {
                    loadSample({ id: activeSampleId, hint: languageHint })
                  } else if (selectedFile) {
                    runExtraction(selectedFile, selectedFile.name, languageHint)
                  }
                }}
              >
                {loading ? <div className="spinner"></div> : <Play size={14} fill="#fff" />}
                <span>{loading ? 'Processing...' : 'Run Pipeline'}</span>
              </button>
            </div>
          </div>

          {/* Right Column: Key-Value Cards, Telemetry, JSON Inspector */}
          <div className="glass-panel results-card">
            <div className="card-header-bar">
              <div className="panel-title">
                <CheckCircle2 size={16} color="#10b981" />
                <span>EXTRACTED IDENTITY FIELDS</span>
              </div>

              {result?.language_model_used && (
                <div className="badge badge-emerald">
                  {result.language_model_used}
                </div>
              )}
            </div>

            {/* CPU Telemetry Strip */}
            <LatencyGauge latency={result?.latency} />

            {/* Structured Fields Display */}
            <KeyValueCards
              fields={result?.fields}
              documentType={result?.document_type}
              modelUsed={null}
            />

            {/* Raw JSON viewer */}
            {result && <JsonViewer data={result} />}
          </div>
        </div>
      </main>

      <footer
        style={{
          marginTop: 'auto',
          padding: '1rem',
          textAlign: 'center',
          borderTop: '1px solid var(--border-dim)',
          color: 'var(--text-dim)',
          fontSize: '0.75rem',
          fontFamily: 'var(--font-mono)',
        }}
      >
        OCR Engine • Offline ONNX Runtime CPU Pipeline
      </footer>
    </div>
  )
}
