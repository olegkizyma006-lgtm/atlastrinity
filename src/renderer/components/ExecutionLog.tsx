/**
 * ExecutionLog - Left panel log display
 * Cyberpunk Terminal Style
 */

import * as React from 'react';
import { useRef, useLayoutEffect, useState, useEffect, useCallback } from 'react';

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
}

const ExecutionLog: React.FC<ExecutionLogProps> = ({ logs }) => {
  // Filter out noisy connection logs
  const filteredLogs = logs.filter(
    (l) => !l.message.includes('Connected to') && !l.message.includes('health check')
  );

  const logsEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Track if user has scrolled away from bottom (to pause auto-scroll)
  const [userScrolledUp, setUserScrolledUp] = useState(false);
  const lastLogCountRef = useRef(filteredLogs.length);

  // Check if user is near bottom
  const isNearBottom = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return true;
    const { scrollTop, scrollHeight, clientHeight } = container;
    // Using a more robust threshold and Math.ceil for fractional values
    return Math.ceil(scrollHeight - scrollTop - clientHeight) <= 50;
  }, []);

  // Handle scroll events to detect user scrolling
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      if (isNearBottom()) {
        setUserScrolledUp(false);
      }
    };

    const handleWheel = (e: WheelEvent) => {
      // Any scroll action by user should pause auto-scroll if it moves away from bottom
      if (e.deltaY < 0) {
        setUserScrolledUp(true);
      }
      
      // Resumes auto-scroll if user scrolls down and IS near bottom
      if (e.deltaY > 0 && isNearBottom()) {
        setUserScrolledUp(false);
      }
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    container.addEventListener('wheel', handleWheel, { passive: true });

    return () => {
      container.removeEventListener('scroll', handleScroll);
      container.removeEventListener('wheel', handleWheel);
    };
  }, [isNearBottom]);

  // Auto-scroll logic - only scroll if user hasn't scrolled up
  useLayoutEffect(() => {
    const hasNewLogs = filteredLogs.length > lastLogCountRef.current;
    lastLogCountRef.current = filteredLogs.length;

    // Auto-scroll logic:
    // 1. If we are already near bottom
    // 2. If it's the very first log(s)
    // 3. If new logs arrived AND user hasn't explicitly scrolled up
    if (isNearBottom() || filteredLogs.length <= 1 || (hasNewLogs && !userScrolledUp)) {
      // Use a small timeout to ensure DOM has rendered
      const timer = setTimeout(() => {
        if (logsEndRef.current) {
          logsEndRef.current.scrollIntoView({ behavior: 'auto', block: 'end' });
        }
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [filteredLogs, userScrolledUp, isNearBottom]);

  const formatTime = (ts: Date | number | string) => {
    const d = ts instanceof Date ? ts : new Date(Number(ts) * 1000);
    return d.toLocaleTimeString([], {
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
    <div className="flex-1 flex flex-col h-full overflow-hidden font-mono relative min-h-0">
      <div style={{ height: '32px' }} /> {/* Spacer for title bar area */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto p-1 scrollbar-thin min-h-0 pb-24"
      >
        {filteredLogs.map((log) => (
          <div
            key={log.id}
            className="flex flex-col mb-2 animate-fade-in group hover:bg-white/5 px-1 py-1 rounded transition-colors"
          >
            <div className="flex items-center mb-1">
              <div className="flex items-center gap-4 filter grayscale opacity-20 group-hover:grayscale-0 group-hover:opacity-40 transition-all duration-500">
                <span
                  className="text-[8px] font-bold tracking-[0.2em] uppercase"
                  style={{
                    color:
                      log.agent === 'GRISHA'
                        ? 'var(--grisha-orange)'
                        : log.agent === 'TETYANA'
                          ? 'var(--tetyana-green)'
                          : log.agent === 'USER'
                            ? 'var(--user-turquoise)'
                            : 'var(--atlas-blue)',
                    fontFamily: 'JetBrains Mono',
                  }}
                >
                  {log.agent}
                </span>

                <div
                  className="flex items-center gap-3 text-[8px] font-mono font-medium tracking-[0.05em] uppercase"
                  style={{
                    color: getLogColor(log.type),
                  }}
                >
                  <span className="tracking-tighter">{formatTime(log.timestamp)}</span>
                  <span className="font-bold">{log.type.toUpperCase()}</span>
                </div>
              </div>
            </div>

            {/* Content Row */}
            <div className="flex-1 flex flex-col pl-0.5">
              {/* Message */}
              <span
                className={`text-[11px] font-normal leading-relaxed break-words transition-colors font-mono ${
                  log.message.includes('[VIBE-THOUGHT]')
                    ? 'text-gray-400 pl-4 italic ml-2 border-l border-gray-700/50'
                    : log.message.includes('[VIBE-ACTION]')
                      ? 'text-yellow-400'
                      : log.message.includes('[VIBE-GEN]')
                        ? 'text-green-400'
                        : log.message.includes('[VIBE-LIVE]')
                          ? 'text-blue-300'
                          : log.agent === 'USER'
                            ? 'text-[#00E5FF]'
                            : 'text-[#00A3FF] group-hover:text-[#33B5FF]'
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

        {filteredLogs.length === 0 && (
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
