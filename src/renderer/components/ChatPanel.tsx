/**
 * ChatPanel - Right panel for agent messages
 */

import * as React from 'react';
import { useEffect, useRef } from 'react';

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

  const scrollToBottom = () => {
    if (scrollContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 150;
      if (isAtBottom) {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }
    } else {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [filteredMessages]);

  const getAgentColor = (agent: string) => {
    const a = agent.toUpperCase().trim();
    switch (a) {
      case 'USER':
        return 'var(--user-turquoise, #00E5FF)';
      default:
        // All agents and system go blue
        return 'var(--atlas-blue, #00A3FF)';
    }
  };

  return (
    <div className="flex-1 flex flex-col p-4 font-mono h-full overflow-hidden relative">
      <div style={{ height: '32px' }} /> {/* Spacer for title bar area */}

      {/* Main Chat Stream */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto pr-1 scrollbar-thin h-full"
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
                      style={{ color: getAgentColor(msg.agent), fontFamily: 'Outfit' }}
                    >
                      {msg.agent}
                    </span>
                    <span
                      className="text-[6.5px] font-mono tracking-tighter uppercase font-medium"
                      style={{ color: getAgentColor(msg.agent) }}
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
                  className="text-[10px] font-[200] leading-relaxed break-words pl-0.5 py-0.5 message-text transition-colors"
                  style={{ fontFamily: 'Outfit', letterSpacing: '0.01em' }}
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
