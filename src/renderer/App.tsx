/**
 * AtlasTrinity - Main App Component
 * Premium Design System Integration
 */

import React, { useState, useEffect } from 'react';
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
}

interface SystemMetrics {
  cpu: string;
  memory: string;
  net_up_val: string;
  net_up_unit: string;
  net_down_val: string;
  net_down_unit: string;
}

const App: React.FC = () => {
  const [systemState, setSystemState] = useState<SystemState>('IDLE');
  const [activeAgent, setActiveAgent] = useState<AgentName>('ATLAS');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);
  const [activeMode, setActiveMode] = useState<'STANDARD' | 'LIVE'>('STANDARD');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
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

  // Initialize & Poll State
  useEffect(() => {
    const pollState = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/state');
        if (response.ok) {
          const data = await response.json();

          // Sync system state
          setSystemState(data.system_state || 'IDLE');
          setActiveAgent(data.active_agent || 'ATLAS');
          setCurrentTask(data.current_task || '');
          setActiveMode(data.active_mode || 'STANDARD');
          if (data.metrics) setMetrics(data.metrics);

          if (data.logs && data.logs.length > 0) {
            const newLogs = data.logs.map(
              (l: {
                agent: AgentName;
                message: string;
                type: LogEntry['type'];
                timestamp: number;
              }) => ({
                ...l,
                id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                timestamp: new Date(l.timestamp * 1000),
              })
            );
            setLogs(newLogs);
          }

          if (data.messages && data.messages.length > 0) {
            setChatHistory(
              data.messages.map((m: { agent: AgentName; text: string; timestamp: number }) => ({
                ...m,
                timestamp: new Date(m.timestamp * 1000),
              }))
            );
          }
        }
      } catch (err) {
        // Silent fail to avoid flooding console if server is starting up
      }
    };

    pollState();
    const interval = setInterval(pollState, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleCommand = async (cmd: string) => {
    // 1. Log user action
    addLog('ATLAS', `Command: ${cmd}`, 'action');
    setSystemState('PROCESSING');
    try {
      const response = await fetch('http://127.0.0.1:8000/api/chat', {
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

  // Derived messages for ChatPanel
  const chatMessages = chatHistory.map((m) => ({
    id: `chat-${m.timestamp.getTime()}-${Math.random()}`,
    agent: m.agent,
    text: m.text,
    timestamp: m.timestamp,
  }));

  return (
    <div className="app-container scanlines">
      {/* Pulsing Borders */}
      <div className="pulsing-border top"></div>
      <div className="pulsing-border bottom"></div>
      <div className="pulsing-border left"></div>
      <div className="pulsing-border right"></div>

      {/* Left Panel - Execution Log */}
      <div className="panel left-panel glass-panel">
        <div className="flex-1 flex flex-col p-2 overflow-hidden">
          <ExecutionLog logs={logs} />
        </div>
      </div>

      {/* Center Panel - Neural Core Visualization */}
      <div className="panel center-panel">
        {/* Status Overlay - REMOVED FROM TOP */}
        <div className="fixed top-0 left-0 w-full flex justify-center z-[100] pointer-events-none"></div>

        {/* Main Core */}
        <NeuralCore state={systemState} activeAgent={activeAgent} />
      </div>

      {/* Right Panel - Chat */}
      <div className="panel right-panel glass-panel">
        <div className="right-panel-body">
          {/* Chat Area - Flexible Height */}

          <div className="chat-panel-container">
            <ChatPanel messages={chatMessages} />
          </div>
        </div>
      </div>

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
