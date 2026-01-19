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
  onNewSession?: () => void;
}

const ChatPanel: React.FC<ChatPanelProps> = ({ messages, onNewSession }) => {
  // STRICT FILTER: Only voice messages as requested
  const filteredMessages = messages.filter((m) => m.type === 'voice');

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
      case 'GRISHA':
        return 'var(--grisha-orange, #FFB800)';
      case 'TETYANA':
        return 'var(--tetyana-green, #00FF88)';
      case 'USER':
        return '#FFFFFF';
      case 'SYSTEM':
        return 'var(--atlas-blue, #00A3FF)';
      default:
        return 'var(--atlas-blue, #00A3FF)';
    }
  };

  return (
    <div className="flex-1 flex flex-col p-4 font-mono h-full overflow-hidden relative">
      {/* Absolute Positioned Header to align with top line */}
      <div className="absolute top-[-38px] right-0 flex items-center gap-3 opacity-30 hover:opacity-100 transition-opacity shrink-0 uppercase tracking-[0.4em] text-[6px] font-bold select-none cursor-default">
        <div className="flex items-center gap-1.5">
          <div className="w-[5px] h-[5px] rounded-full border border-white/20"></div>
          <span>communication::hud</span>
        </div>

        {/* New Session Plus Button */}
        <button
          onClick={onNewSession}
          className="bg-white/5 hover:bg-white/20 border border-white/10 rounded px-1.5 py-0.5 transition-colors flex items-center justify-center group"
          title="New Session"
        >
          <span className="text-[10px] leading-none group-hover:scale-110 transition-transform">+</span>
        </button>
      </div>

      {/* Main Chat Stream */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto pr-1 scrollbar-thin"
        style={{ height: '0', minHeight: '100%' }}
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
                  className="text-[8.5px] font-[200] leading-relaxed break-words pl-0.5 py-0.5 text-white/50 group-hover:text-white/85 transition-colors"
                  style={{ fontFamily: 'Outfit', letterSpacing: '0.02em' }}
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
