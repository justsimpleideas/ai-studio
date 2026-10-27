import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// Tokens, then global styles, so the image side's scoped rules (imported by App) win ties.
import './styles/tokens.css'
import './styles/app.css'
import { App } from './App.js'

const container = document.getElementById('root')
if (!container) throw new Error('Missing #root')

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>
)
