import { AIDashboard } from '../components/ai-dashboard/AIDashboard'

export function Dashboard() {
  return <AIDashboard view="overview" />
}

export function AISignalsPage() {
  return <AIDashboard view="signals" />
}

export function AIPerformancePage() {
  return <AIDashboard view="performance" />
}

export function AICalibrationPage() {
  return <AIDashboard view="calibration" />
}

export function AISystemPage() {
  return <AIDashboard view="system" />
}
