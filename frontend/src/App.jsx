import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import FindJobs from './pages/FindJobs'
import ResumeCenter from './pages/ResumeCenter'
import AtsChecker from './pages/AtsChecker'
import CareerRoadmap from './pages/CareerRoadmap'
import InterviewCoach from './pages/InterviewCoach'
import CoPilot from './pages/CoPilot'
import JobTracker from './pages/JobTracker'
import EasyApply from './pages/EasyApply'
import Profile from './pages/Profile'

function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-gray-50">
        <Sidebar />
        <main className="flex-1 ml-64 p-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/find-jobs" element={<FindJobs />} />
            <Route path="/resume-center" element={<ResumeCenter />} />
            <Route path="/ats-checker" element={<AtsChecker />} />
            <Route path="/career-roadmap" element={<CareerRoadmap />} />
            <Route path="/interview-coach" element={<InterviewCoach />} />
            <Route path="/copilot" element={<CoPilot />} />
            <Route path="/job-tracker" element={<JobTracker />} />
            <Route path="/easy-apply" element={<EasyApply />} />
            <Route path="/profile" element={<Profile />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
