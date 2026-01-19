/**
 * ExecutionLog - Left panel log display
 * Cyberpunk Terminal Style
 */

import * as React from 'react';
import { useRef, useEffect } from 'react';

type AgentName = 'ATLAS' | 'TETYANA' | 'GRISHA' | 'SYSTEM' | 'USER';

interface LogEntry {
  id: string;
  timestamp: Date;
  agent: AgentName;
  message: string;
  type: 'info' | 'action' | 'success' | 'error' | 'voice';
}

interface ExecutionLogProps {
  logs: LogEntry[];
  onNewSession?: () => void;
  onToggleHistory?: () => void;
}

const ExecutionLog: React.FC<ExecutionLogProps> = ({ logs, onNewSession, onToggleHistory }) => {
  // Filter out noisy connection logs
  const filteredLogs = logs.filter(
    (l) => !l.message.includes('Connected to') && !l.message.includes('health check')
  );

  const logsEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (scrollContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 150; // Changed threshold from 100 to 150
      if (isAtBottom) {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }
    } else {
      logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [filteredLogs]);

  const formatTime = (ts: number) => {
    return new Date(ts * 1000).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  };

  const getLogColor = (type: string) => {
    switch (type) {
      case 'error':
        return '#FF4D4D';
      case 'warning':
        return '#FFB800';
      case 'success':
        return '#00FF88';
      case 'action':
        return '#00A3FF';
      default:
        return 'rgba(255, 255, 255, 0.5)';
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden font-mono relative">
      {/* Window Header - Absolute Positioned to align with traffic lights */}
      <div className="absolute top-[-38px] left-[110px] right-0 flex items-center justify-between gap-1.5 opacity-30 hover:opacity-100 transition-opacity shrink-0 select-none px-4">
        <span className="text-[6px] tracking-[0.4em] uppercase font-bold text-white/50">
          core::log_stream
        </span>

        <div className="flex items-center gap-1.5">
          {/* History Button */}
          <button
            onClick={onToggleHistory}
            className="control-btn"
            style={{ width: '22px', height: '22px', padding: '4px' }}
            title="Session History"
          >
            <span className="text-[9px] leading-none group-hover:scale-110 transition-transform">⌛</span>
          </button>

          {/* New Session Button */}
          <button
            onClick={onNewSession}
            className="control-btn"
            style={{ width: '22px', height: '22px', padding: '4px' }}
            title="New Session"
          >
            <span className="text-[12px] leading-none group-hover:scale-110 transition-transform">+</span>
          </button>
        </div>
      </div>

      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto p-1 scrollbar-thin"
        style={{ height: '0', minHeight: '100%' }}
      >
        {filteredLogs.map((log) => (
          <div
            key={log.id}
            className="flex flex-col mb-2 animate-fade-in group hover:bg-white/5 px-1 py-1 rounded transition-colors"
          >
            <div className="flex items-center mb-1">
              <div className="flex items-center gap-4 filter grayscale opacity-20 group-hover:grayscale-0 group-hover:opacity-40 transition-all duration-500">
                <span
                  className="text-[6.5px] font-medium tracking-[0.2em] uppercase"
                  style={{
                    color:
                      log.agent === 'GRISHA'
                        ? 'var(--grisha-orange)'
                        : log.agent === 'TETYANA'
                          ? 'var(--tetyana-green)'
                          : log.agent === 'USER'
                            ? '#FFFFFF'
                            : 'var(--atlas-blue)',
                    fontFamily: 'Outfit',
                  }}
                >
                  {log.agent}
                </span>

                <div
                  className="flex items-center gap-3 text-[6.5px] font-mono font-medium tracking-[0.05em] uppercase"
                  style={{
                    color: getLogColor(log.type),
                  }}
                >
                  <span className="tracking-tighter">{formatTime(Number(log.timestamp))}</span>
                  <span className="font-bold">{log.type.toUpperCase()}</span>
                </div>
              </div>
            </div>

            {/* Content Row */}
            <div className="flex-1 flex flex-col pl-0.5">
              {/* Message */}
              <span
                className={`text-[8.5px] font-light leading-relaxed break-words transition-colors font-mono ${log.message.includes('[VIBE-THOUGHT]')
                  ? 'text-gray-400 pl-4 italic ml-2 border-l border-gray-700/50'
                  : log.message.includes('[VIBE-ACTION]')
                    ? 'text-yellow-400'
                    : log.message.includes('[VIBE-GEN]')
                      ? 'text-green-400'
                      : log.message.includes('[VIBE-LIVE]')
                        ? 'text-blue-300'
                        : 'text-white/50 group-hover:text-white/85'
                  }`}
                style={{ fontFamily: 'JetBrains Mono' }}
              >
                {typeof log.message === 'object'
                  ? JSON.stringify(log.message)
                  : log.message.replace('🧠 [VIBE-THOUGHT]', '').trim()}
              </span>
            </div>
          </div>
        ))}

        <div ref={logsEndRef} />

        {logs.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center opacity-10 text-[9px] gap-2 tracking-[0.4em] uppercase">
            <div className="w-10 h-10 rounded-full border border-current animate-spin-slow opacity-20"></div>
            <span>System Initialized</span>
            <span>Awaiting Core Link...</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ExecutionLog;
