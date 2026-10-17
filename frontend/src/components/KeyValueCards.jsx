import React, { useState } from 'react'
import { Copy, Check, User, Hash, Calendar, Shield, Globe, Award, Fingerprint } from 'lucide-react'

export default function KeyValueCards({ fields, documentType, modelUsed }) {
  const [copiedKey, setCopiedKey] = useState(null)

  const handleCopy = (key, text) => {
    if (!text) return
    navigator.clipboard.writeText(text)
    setCopiedKey(key)
    setTimeout(() => setCopiedKey(null), 1800)
  }

  const renderField = (key, label, icon, fieldData, fullWidth = false) => {
    const val = fieldData?.value
    const conf = fieldData?.confidence ? Math.round(fieldData.confidence * 100) : 0
    const source = fieldData?.source || 'VIZ'
    const isCopied = copiedKey === key

    return (
      <div className={`field-box ${fullWidth ? 'full-width' : ''}`}>
        <div className="field-label">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            {icon}
            <span>{label}</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            {val && (
              <span className={`badge ${conf >= 80 ? 'badge-emerald' : 'badge-amber'}`}>
                {conf}% {source}
              </span>
            )}
            {val && (
              <button
                className="copy-btn"
                onClick={() => handleCopy(key, val)}
                title="Copy value"
              >
                {isCopied ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
              </button>
            )}
          </div>
        </div>

        <div className={`field-value ${!val ? 'empty' : ''}`}>
          {val || 'Not detected'}
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Document Classification Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div>
          <span style={{ fontSize: '0.72rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Classification
          </span>
          <div style={{ fontSize: '1.15rem', fontWeight: '700', color: '#fff', fontFamily: 'var(--font-display)' }}>
            {documentType || 'Identity Document'}
          </div>
        </div>

        {modelUsed && (
          <div className="badge badge-emerald" style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}>
            {modelUsed}
          </div>
        )}
      </div>

      {/* Grid of Key-Value Cards */}
      <div className="fields-grid">
        {renderField('name', 'Full Name', <User size={14} color="#818cf8" />, fields?.full_name, true)}
        {renderField('doc_num', 'Document Number', <Hash size={14} color="#38bdf8" />, fields?.document_number)}
        {renderField('dob', 'Date of Birth', <Calendar size={14} color="#f472b6" />, fields?.date_of_birth)}
        {renderField('expiry', 'Expiry Date', <Shield size={14} color="#fbbf24" />, fields?.expiry_date)}
        {renderField('nat', 'Nationality', <Globe size={14} color="#34d399" />, fields?.nationality)}
        {renderField('sex', 'Sex / Gender', <Award size={14} color="#c084fc" />, fields?.sex)}
        {fields?.personal_number?.value &&
          renderField('personal_no', 'Personal Number (CNP / PIN)', <Fingerprint size={14} color="#a78bfa" />, fields.personal_number, true)
        }
      </div>
    </div>
  )
}
