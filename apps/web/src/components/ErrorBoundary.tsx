import { Component, type ErrorInfo, type ReactNode } from 'react'
import { RefreshCw, TriangleAlert } from 'lucide-react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

/** Last-resort catch for render-time exceptions anywhere below it. Without this, a
    single component throwing turns the whole app into a blank white screen with no
    way back short of the browser's own reload. Deliberately dependency-light — no
    i18n, no app state — so the fallback itself can't be taken down by whatever broke
    the tree it's catching. One boundary at the root is enough for this app's size;
    it doesn't need per-section boundaries to isolate failures from each other. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // eslint-disable-next-line no-console -- the one place this app intentionally logs to console
    console.error('Unhandled render error:', error, info.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="flex min-h-screen items-center justify-center bg-canvas p-6">
        <div className="w-full max-w-md rounded-xl border border-line-strong bg-raised p-6 shadow-raised">
          <div className="flex items-start gap-3">
            <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-danger-soft">
              <TriangleAlert size={16} className="text-danger-ink" aria-hidden />
            </span>
            <div className="min-w-0">
              <h1 className="text-title font-semibold text-ink">Something went wrong</h1>
              <p className="mt-1 text-sm text-ink-2">
                This view hit an unexpected error. Your uploaded documents and settings are
                unaffected — reloading will bring you back to where you were.
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="mt-4 flex w-full items-center justify-center gap-1.5 rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition-colors duration-100 hover:bg-primary-strong"
          >
            <RefreshCw size={14} aria-hidden />
            Reload
          </button>
          <details className="mt-3 text-2xs text-ink-3">
            <summary className="cursor-pointer select-none">Technical details</summary>
            <pre className="mt-1.5 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-rail/50 p-2 font-mono">
              {this.state.error.message}
              {this.state.error.stack ? `\n\n${this.state.error.stack}` : ''}
            </pre>
          </details>
        </div>
      </div>
    )
  }
}
