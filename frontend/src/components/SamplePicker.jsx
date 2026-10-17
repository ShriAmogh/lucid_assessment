import React from 'react'
import { FileText } from 'lucide-react'

export const SAMPLES = [
  { id: 'roman_id_1.png', label: 'Romanian ID', sub: 'Carte de Identitate', flag: 'RO', hint: 'romanian' },
  { id: 'german_id_1.png', label: 'German ID', sub: 'Personalausweis', flag: 'DE', hint: 'german' },
  { id: 'greek_id_1.png', label: 'Greek ID', sub: 'Δελτίο Ταυτότητας', flag: 'GR', hint: 'greek' },
  { id: 'english_id_1.png', label: 'UK Driver License', sub: 'Great Britain DVLA', flag: 'UK', hint: 'english' },
  { id: 'german_id_2.png', label: 'German Driving License', sub: 'Führerschein', flag: 'DE', hint: 'german' },
  { id: 'greek_id_2.png', label: 'Greek ID (Sample 2)', sub: 'Δελτίο Ταυτότητας', flag: 'GR', hint: 'greek' },
  { id: 'roman_id_2.png', label: 'Romanian Driving License', sub: 'Permis Conducere', flag: 'RO', hint: 'romanian' },
  { id: 'english_id_2.png', label: 'UK Identity Doc', sub: 'UK ID Card', flag: 'UK', hint: 'english' },
]

export default function SamplePicker({ activeSampleId, onSelectSample, loading }) {
  return (
    <div className="preset-toolbar">
      <div className="preset-toolbar-label">
        <FileText size={15} color="#38bdf8" />
        <span>DOCUMENT SAMPLES:</span>
      </div>

      <div className="preset-buttons">
        {SAMPLES.map((s) => {
          const isActive = activeSampleId === s.id
          return (
            <button
              key={s.id}
              className={`preset-btn ${isActive ? 'active' : ''}`}
              onClick={() => onSelectSample(s)}
              disabled={loading}
              title={`${s.label} (${s.sub})`}
            >
              <span className="preset-flag">{s.flag}</span>
              <span className="preset-label">{s.label}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
