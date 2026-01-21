/**
 * ChatPanel - Right panel for agent messages
 */

import * as React from 'react';
import { useLayoutEffect, useRef, useState, useEffect, useCallback } from 'react';

type AgentName = 'ATLAS' | 'TETYANA' | 'GRISHA' | 'SYSTEM' | 'USER';

interface Message {
  id: string;
  agent: AgentName;
  text: string;
  timestamp: Date;
  type?: 'text' | 'voice';
}

interface ChatPanelProps {
  messages: Message[];
}

const ChatPanel: React.FC<ChatPanelProps> = ({ messages }) => {
  // FILTER: Only voice messages and user messages
  const filteredMessages = messages.filter((m) => m.type === 'voice' || m.agent === 'USER');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  
  // Track if user has scrolled away from bottom (to pause auto-scroll)
  const [userScrolledUp, setUserScrolledUp] = useState(false);
  const lastMessageCountRef = useRef(filteredMessages.length);

  // Check if user is near bottom
  const isNearBottom = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return true;
    const { scrollTop, scrollHeight, clientHeight } = container;
    return scrollHeight - scrollTop - clientHeight < 100;
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
      // User scrolling up = pause auto-scroll
      if (e.deltaY < 0 && !isNearBottom()) {
        setUserScrolledUp(true);
      }
      // User scrolling down and near bottom = resume
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
    const container = scrollContainerRef.current;
    if (!container) return;

    const hasNewMessages = filteredMessages.length > lastMessageCountRef.current;
    lastMessageCountRef.current = filteredMessages.length;

    // Auto-scroll if: near bottom, OR new message arrived and user hasn't scrolled up, OR first messages
    if (isNearBottom() || (hasNewMessages && !userScrolledUp) || filteredMessages.length <= 1) {
      container.scrollTop = container.scrollHeight;
    }
  }, [filteredMessages, userScrolledUp, isNearBottom]);

  const getHeaderColor = (agent: string) => {
    const a = agent.toUpperCase().trim();
    switch (a) {
      case 'GRISHA':
        return 'var(--grisha-orange, #FFB800)';
      case 'TETYANA':
        return 'var(--tetyana-green, #00FF88)';
      case 'USER':
        return 'var(--user-turquoise, #00E5FF)';
      default:
        return 'var(--atlas-blue, #00A3FF)';
    }
  };

  const getMessageTextColor = (agent: string) => {
    return agent.toUpperCase().trim() === 'USER'
      ? 'var(--user-turquoise, #00E5FF)'
      : 'var(--atlas-blue, #00A3FF)';
  };

  return (
    <div className="flex-1 flex flex-col p-4 font-mono h-full overflow-hidden relative min-h-0">
      <div style={{ height: '32px' }} /> {/* Spacer for title bar area */}
      {/* Main Chat Stream */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto pr-1 scrollbar-thin h-full min-h-0"
      >
        {filteredMessages.length === 0 ? (
          <div className="h-full flex items-center justify-center opacity-10 italic text-[9px] tracking-[0.5em] uppercase">
            Waiting for neural link...
          </div>
        ) : (
          <div className="flex flex-col gap-2 py-1">
            {filteredMessages.map((msg) => (
              <div key={msg.id} className="animate-fade-in group mb-3">
                <div className="flex items-center mb-1.5">
                  <div className="flex items-center gap-4 filter grayscale opacity-20 group-hover:grayscale-0 group-hover:opacity-40 transition-all duration-500">
                    <span
                      className="text-[6.5px] font-medium tracking-[0.1em] uppercase"
                      style={{ color: getHeaderColor(msg.agent), fontFamily: 'Outfit' }}
                    >
                      {msg.agent}
                    </span>
                    <span
                      className="text-[6.5px] font-mono tracking-tighter uppercase font-medium"
                      style={{ color: getHeaderColor(msg.agent) }}
                    >
                      {msg.timestamp.toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                      })}
                    </span>
                  </div>
                </div>

                <div
                  className="text-[10px] font-[300] leading-relaxed break-words pl-0.5 py-0.5 message-text transition-colors"
                  style={{
                    fontFamily: 'Outfit',
                    letterSpacing: '0.01em',
                    color: getMessageTextColor(msg.agent),
                  }}
                >
                  {msg.text}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatPanel;
