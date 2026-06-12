import React from 'react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, message: '' };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || 'Unknown error' };
  }

  componentDidCatch(error, info) {
    console.error('[Kiosk ErrorBoundary]', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="app-shell">
          <div className="app-main">
            <div className="error-fallback card screen">
              <h2>Something went wrong</h2>
              <p style={{ color: 'var(--text-muted)', margin: '16px 0' }}>
                {this.state.message}
              </p>
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => {
                  this.setState({ hasError: false, message: '' });
                  window.location.href = '/';
                }}
              >
                Return to Welcome
              </button>
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
