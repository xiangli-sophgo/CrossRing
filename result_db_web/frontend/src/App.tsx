import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from 'antd';
import ExperimentList from './pages/ExperimentList';
import ExperimentDetail from './pages/ExperimentDetail';
import CompareView from './pages/CompareView';
import ErrorBoundary from './components/ErrorBoundary';

const { Content } = Layout;

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Layout style={{ minHeight: '100vh' }}>
          <Content style={{ padding: '24px', background: '#f0f2f5' }}>
            <Routes>
              <Route path="/" element={<ExperimentList />} />
              <Route path="/experiments/:id" element={<ExperimentDetail />} />
              <Route path="/compare" element={<CompareView />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Content>
        </Layout>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
