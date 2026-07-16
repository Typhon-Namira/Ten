import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Router } from './router/Router'
import './styles.css'

createRoot(document.getElementById('root')!).render(<StrictMode><Router /></StrictMode>)
