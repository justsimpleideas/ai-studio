import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// Tokens, then global styles, so the image side's scoped rules (imported by App) win ties.
import './styles/tokens.css'
import './styles/app.css'
import { App } from './App.js'

// The theme the main process loaded the page with, set before the first paint; Settings
// changes it afterwards through the config (App, useTheme).
const theme = new URLSearchParams(location.search).get('theme')
if (theme === 'light') document.documentElement.dataset['theme'] = 'light'

const container = document.getElementById('root')
if (!container) throw new Error('Missing #root')

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>
)
