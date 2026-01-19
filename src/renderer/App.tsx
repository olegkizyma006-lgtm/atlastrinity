/**
 * AtlasTrinity - Main App Component
 * Premium Design System Integration
 */

import * as React from 'react';
import { useState, useEffect } from 'react';
import NeuralCore from './components/NeuralCore';
import ExecutionLog from './components/ExecutionLog.tsx';
import AgentStatus from './components/AgentStatus.tsx';
import ChatPanel from './components/ChatPanel.tsx';
import CommandLine from './components/CommandLine.tsx';

// Agent types
type AgentName = 'ATLAS' | 'TETYANA' | 'GRISHA' | 'SYSTEM' | 'USER';
type SystemState = 'IDLE' | 'PROCESSING' | 'EXECUTING' | 'VERIFYING' | 'ERROR';

interface LogEntry {
  id: string;
  timestamp: Date;
  agent: AgentName;
  message: string;
  type: 'info' | 'action' | 'success' | 'error' | 'voice';
}

interface ChatMessage {
  agent: AgentName;
  text: string;
  timestamp: Date;
  type?: 'text' | 'voice';
}

interface SystemMetrics {
  cpu: string;
  memory: string;
  net_up_val: string;
  net_up_unit: string;
  net_down_val: string;
  net_down_unit: string;
}

const API_BASE = 'http://127.0.0.1:8000';

const App: React.FC = () => {
  const [systemState, setSystemState] = useState<SystemState>('IDLE');
  const [activeAgent, setActiveAgent] = useState<AgentName>('ATLAS');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);
  const [activeMode, setActiveMode] = useState<'STANDARD' | 'LIVE'>('STANDARD');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [sessions, setSessions] = useState<any[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('current_session');
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [metrics, setMetrics] = useState<SystemMetrics>({
    cpu: '0%',
    memory: '0.0GB',
    net_up_val: '0.0',
    net_up_unit: 'K/S',
    net_down_val: '0.0',
    net_down_unit: 'K/S',
  });

  // Add log entry
  const addLog = (agent: AgentName, message: string, type: LogEntry['type'] = 'info') => {
    const entry: LogEntry = {
      id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      agent,
      message,
      type,
    };
    setLogs((prev) => [...prev.slice(-100), entry]); // Keep last 100 entries
  };

  const [currentTask, setCurrentTask] = useState<string>('');

  const fetchSessions = async (retryCount = 0) => {
    try {
      const response = await fetch(`${API_BASE}/api/sessions`);
      if (!response.ok) throw new Error('Failed to fetch');
      const data = await response.json();
      setSessions(data);
    } catch (err) {
      if (retryCount < 5) {
        // Retry with exponential backoff if server is still starting
        const delay = Math.pow(2, retryCount) * 1000;
        console.warn(`[BRAIN] Session fetch failed, retrying in ${delay}ms... (Attempt ${retryCount + 1}/5)`);
        setTimeout(() => fetchSessions(retryCount + 1), delay);
      } else {
        console.error('Failed to fetch sessions after retries:', err);
      }
    }
  };

  const pollState = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/state`);
      if (response.ok) {
        const data = await response.json();
        if (data) {
          // Sync system state
          setSystemState(data.system_state || 'IDLE');
          setActiveAgent(data.active_agent || 'ATLAS');
          if (data.session_id) setCurrentSessionId(data.session_id);
          setCurrentTask(data.current_task || '');
          setActiveMode(data.active_mode || 'STANDARD');
          if (data.metrics) setMetrics(data.metrics);

          if (data.logs) {
            setLogs(data.logs); // Keep as numbers
          }

          if (data.messages) {
            setChatHistory(
              data.messages.map(
                (m: { agent: AgentName; text: string; timestamp: number; type: 'text' | 'voice' }) => ({
                  ...m,
                  timestamp: new Date(m.timestamp * 1000),
                })
              )
            );
          }
        }
      }
    } catch (err) {
      if (err instanceof TypeError && err.message === 'Failed to fetch') {
        const now = Date.now();
        // Only log connection refused if it persists beyond initial startup (5s)
        if (!window.hasOwnProperty('startTime')) (window as any).startTime = now;
        if (now - (window as any).startTime > 5000) {
          console.warn(`[BRAIN] Connection refused. Is the Python server running on ${API_BASE}?`);
        }
      } else {
        console.error('[BRAIN] Polling error:', err);
      }
    }
  };

  // Initialize & Poll State
  useEffect(() => {
    pollState();
    fetchSessions();
    const interval = setInterval(pollState, 1500); // Polling every 1.5s
    return () => clearInterval(interval);
  }, []);

  const handleCommand = async (cmd: string) => {
    // 1. Log user action
    addLog('ATLAS', `Command: ${cmd}`, 'action');
    setSystemState('PROCESSING');
    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request: cmd }),
      });

      if (!response.ok) throw new Error(`Server Error: ${response.status}`);

      const data = await response.json();

      // 3. Handle Result
      if (data.status === 'completed') {
        const result = data.result;
        // Safely handle object results by stringifying them
        let message = '';
        if (typeof result === 'string') {
          message = result;
        } else if (typeof result === 'object') {
          // Check if it's the specific step results array and format it nicely
          if (Array.isArray(result)) {
            const steps = result.filter((r: { success: boolean }) => r.success).length;
            message = `Task completed successfully: ${steps} steps executed.`;
          } else if (result.result) {
            message =
              typeof result.result === 'string' ? result.result : JSON.stringify(result.result);
          } else {
            message = JSON.stringify(result);
          }
        } else {
          message = String(result);
        }

        addLog('ATLAS', message, 'success');
        setSystemState('IDLE');
      } else {
        addLog('TETYANA', 'Task execution finished', 'info');
        setSystemState('IDLE');
      }
    } catch (error) {
      console.error(error);
      addLog('ATLAS', 'Failed to reach Neural Core. Is Python server running?', 'error');
      setSystemState('ERROR');
    }
  };

  const handleNewSession = async () => {
    console.log('Starting new session...');
    try {
      const response = await fetch(`${API_BASE}/api/session/reset`, {
        method: 'POST',
      });
      if (response.ok) {
        const result = await response.json();
        // Clear local state
        setLogs([]);
        setChatHistory([]);
        if (result.session_id) setCurrentSessionId(result.session_id);
        fetchSessions();
      }
    } catch (err) {
      console.error('Failed to reset session:', err);
    }
  };

  const handleRestoreSession = async (sessionId: string) => {
    console.log(`Restoring session: ${sessionId}`);
    try {
      const response = await fetch(`${API_BASE}/api/sessions/restore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (response.ok) {
        // Clear local state to force refresh from new session
        setLogs([]);
        setChatHistory([]);
        setCurrentSessionId(sessionId);
        setIsHistoryOpen(false);
        pollState();
      }
    } catch (err) {
      console.error('Failed to restore session:', err);
    }
  };

  // Derived messages for ChatPanel
  const chatMessages = chatHistory.map((m) => ({
    id: `chat-${m.timestamp.getTime()}-${Math.random()}`,
    agent: m.agent,
    text: m.text,
    timestamp: m.timestamp,
    type: m.type,
  }));

  return (
    <div className="app-container scanlines">
      {/* Pulsing Borders */}
      <div className="pulsing-border top"></div>
      <div className="pulsing-border bottom"></div>
      <div className="pulsing-border left"></div>
      <div className="pulsing-border right"></div>

      {/* Global Title Bar Controls (Positioned near traffic lights) */}
      <div className="fixed top-2 left-20 z-[100] flex items-center gap-2 pointer-events-auto">
        <button
          onClick={() => setIsHistoryOpen(!isHistoryOpen)}
          className={`titlebar-btn group ${isHistoryOpen ? 'active' : ''}`}
          title="Session History"
        >
          <span className="text-[10px] group-hover:scale-110 transition-transform">⌛</span>
        </button>
        <button
          onClick={handleNewSession}
          className="titlebar-btn group"
          title="New Session"
        >
          <span className="text-[12px] group-hover:scale-110 transition-transform">+</span>
        </button>
      </div>

      {/* Left Panel - Execution Log */}
      <aside className="panel glass-panel left-panel relative">
        <ExecutionLog
          logs={logs}
        />

        {/* Session History Sidebar Overlay */}
        {isHistoryOpen && (
          <div className="absolute inset-0 z-50 bg-[#020202]/95 backdrop-blur-xl border-r border-white/5 animate-slide-in">
            <div className="p-6 h-full flex flex-col">
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-[10px] tracking-[0.4em] uppercase font-bold text-white/40">Session History</h2>
                <button
                  onClick={() => setIsHistoryOpen(false)}
                  className="text-white/30 hover:text-white transition-colors text-xs"
                >
                  ✕
                </button>
              </div>

              <div className="flex-1 overflow-y-auto scrollbar-thin pr-2">
                {sessions.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-[8px] uppercase tracking-widest text-white/10 italic">
                    No history found
                  </div>
                ) : (
                  <div className="flex flex-col gap-3">
                    {sessions.map((s) => (
                      <button
                        key={s.id}
                        onClick={() => handleRestoreSession(s.id)}
                        className={`group p-3 border text-left transition-all duration-300 ${currentSessionId === s.id
                          ? 'bg-white/10 border-white/30'
                          : 'bg-white/5 border-white/5 hover:bg-white/10 hover:border-white/20'
                          }`}
                      >
                        <div className="text-[9px] text-white/80 font-medium mb-1 truncate group-hover:text-white transition-colors">
                          {s.theme}
                        </div>
                        <div className="text-[7px] text-white/30 truncate font-mono">
                          {new Date(s.saved_at).toLocaleString()}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <button
                onClick={handleNewSession}
                className="mt-6 w-full py-3 border border-white/20 bg-white/5 hover:bg-white/10 text-[9px] uppercase tracking-[0.3em] font-bold transition-all"
              >
                + New Session
              </button>
            </div>
          </div>
        )}
      </aside>

      {/* Center Panel: Neural Core */}
      <main className="panel center-panel">
        <NeuralCore state={systemState} activeAgent={activeAgent} />
      </main>

      {/* Right Panel: Chat Panel */}
      <aside className="panel glass-panel right-panel">
        <ChatPanel
          messages={chatMessages}
        />
      </aside>

      {/* Floating Input Dock */}
      <div className="command-dock command-dock-floating">
        <CommandLine
          onCommand={handleCommand}
          isVoiceEnabled={isVoiceEnabled}
          onToggleVoice={() => setIsVoiceEnabled(!isVoiceEnabled)}
        />
      </div>

      {/* Bottom Status Bar - Integrated with AgentStatus */}
      <div className="status-bar !p-0 !bg-transparent !border-none">
        <AgentStatus
          activeAgent={activeAgent}
          systemState={systemState}
          currentTask={currentTask}
          activeMode={activeMode}
          metrics={metrics}
        />
      </div>
    </div>
  );
};

export default App;
