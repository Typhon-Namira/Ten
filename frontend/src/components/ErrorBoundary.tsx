import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertOctagon } from 'lucide-react'

interface Props {
  /** Shown in the fallback so a broken panel is identifiable, e.g. "Decision pipeline". */
  label: string
  children: ReactNode
}

interface State {
  error: Error | null
}

/** Part 3 requirement #3: a panel whose render throws (e.g. an unexpected response shape reaching
 * a `.toFixed()` on `undefined`) must fail visibly and locally, not blank the whole page or read
 * as a misleading "Unavailable" the way a caught, expected empty state does. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`[TEN] "${this.props.label}" failed to render`, error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="panel-render-error" role="alert">
          <AlertOctagon size={18} />
          <div>
            <strong>{this.props.label} failed to render</strong>
            <p>{this.state.error.message}</p>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
