import { Link, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, Briefcase, FileText, CheckCircle, 
  Map, MessageSquare, Bot, ClipboardList, Send, User 
} from 'lucide-react'

const menuItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/find-jobs', label: 'Find Jobs', icon: Briefcase },
  { path: '/resume-center', label: 'Resume Center', icon: FileText },
  { path: '/ats-checker', label: 'ATS Checker', icon: CheckCircle },
  { path: '/career-roadmap', label: 'Career Roadmap', icon: Map },
  { path: '/interview-coach', label: 'Interview Coach', icon: MessageSquare },
  { path: '/copilot', label: 'Co-pilot', icon: Bot },
  { path: '/job-tracker', label: 'Job Tracker', icon: ClipboardList },
  { path: '/easy-apply', label: 'Easy Apply', icon: Send },
  { path: '/profile', label: 'Profile', icon: User },
]

export default function Sidebar() {
  const location = useLocation()

  return (
    <aside className="fixed left-0 top-0 h-full w-64 bg-white border-r border-gray-200 p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-indigo-600">CareerLens AI</h1>
        <p className="text-sm text-gray-500 mt-1">Career Intelligence Platform</p>
      </div>
      <nav className="space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon
          const isActive = location.pathname === item.path
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-indigo-50 text-indigo-600 font-medium'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Icon size={20} />
              <span>{item.label}</span>
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}
