/**
 * AgentStatus - Consolidated Bottom Status Bar
 */

import React from 'react';

type AgentName = 'ATLAS' | 'TETYANA' | 'GRISHA' | 'SYSTEM' | 'USER';
type SystemState = 'IDLE' | 'PROCESSING' | 'EXECUTING' | 'VERIFYING' | 'ERROR';

interface AgentStatusProps {
  activeAgent: AgentName;
  systemState: SystemState;
  currentTask?: string;
  activeMode?: string;
  metrics?: {
    cpu: string;
    memory: string;
    net_up_val: string;
    net_up_unit: string;
    net_down_val: string;
    net_down_unit: string;
  };
}

const AGENT_INFO: Record<AgentName, { ukrainianName: string; color: string }> = {
  ATLAS: { ukrainianName: 'Атлас', color: '#00A3FF' },
  TETYANA: { ukrainianName: 'Тетяна', color: '#00A3FF' },
  GRISHA: { ukrainianName: 'Гріша', color: '#00A3FF' },
  SYSTEM: { ukrainianName: 'Система', color: '#00A3FF' },
  USER: { ukrainianName: 'Користувач', color: '#00E5FF' },
};

const AgentStatus: React.FC<AgentStatusProps> = ({
  activeAgent,
  systemState,
  currentTask,
  activeMode,
  metrics,
}) => {
  const agent = AGENT_INFO[activeAgent];

  const formatTask = (task: string) => {
    if (!task) return 'CORE_IDLE_PREPARING_RESOURCES';
    const goalMatch = task.match(/goal=['"](.*?)['"]/);
    if (goalMatch && goalMatch[1]) return goalMatch[1];
    return task;
  };

  const taskText = formatTask(currentTask || '');

  return (
    <div className="w-full h-6 flex items-center pl-0 pr-4 font-mono text-[8.5px] uppercase tracking-wider overflow-hidden select-none">
      {/* 
          STRICT HUD LAYOUT: 
          All components use fixed-width 'slots' to prevent sub-pixel layout shifts 
          when numerical values change.
      */}
      <div className="flex items-center h-full">
        {/* PHASE SLOT (Total width: 110px) */}
        <div className="flex items-center px-3 h-full gap-2 overflow-hidden">
          <span className="opacity-20 font-light text-[7px] w-[35px] shrink-0">PHASE</span>
          <div className="w-[65px] flex-none">
            <span
              className={`${systemState === 'ERROR' ? 'text-red-500 animate-pulse' : 'text-blue-400'} font-bold block truncate`}
            >
              {systemState}
            </span>
          </div>
        </div>

        {/* AGENT SLOT (Total width: 115px) */}
        <div className="flex items-center px-3 h-full gap-2 border-l border-white/5 overflow-hidden">
          <span className="opacity-20 font-light text-[7px] w-[35px] shrink-0">AGENT</span>
          <div className="w-[65px] flex-none flex items-center gap-1">
            <span style={{ color: agent.color }} className="font-bold truncate grayscale-[0.2]">
              {agent.ukrainianName}
            </span>
            <div className="w-0.5 h-0.5 shrink-0 rounded-full bg-current shadow-[0_0_2px_currentColor]"></div>
          </div>
        </div>

        {/* CPU SLOT (Total width: 75px) */}
        <div className="flex items-center px-3 h-full gap-2 border-l border-white/5 overflow-hidden">
          <span className="opacity-20 font-light text-[7px] w-[22px] shrink-0">CPU</span>
          <div className="w-[30px] flex-none text-right">
            <span className="text-white/90 font-medium tabular-nums">{metrics?.cpu || '0%'}</span>
          </div>
        </div>

        {/* MEM SLOT (Total width: 85px) */}
        <div className="flex items-center px-3 h-full gap-2 border-l border-white/5 overflow-hidden">
          <span className="opacity-20 font-light text-[7px] w-[22px] shrink-0">MEM</span>
          <div className="w-[48px] flex-none text-right">
            <span className="text-white/90 font-medium tabular-nums">
              {metrics?.memory || '0.0GB'}
            </span>
          </div>
        </div>

        {/* NET SLOTS (Total width: ~160px) */}
        <div className="flex items-center px-3 h-full gap-2 border-l border-white/5 overflow-hidden">
          <span className="opacity-20 font-light text-[7px] w-[22px] shrink-0">NET</span>

          {/* UP SLOT */}
          <div className="flex items-center gap-0.5 w-[50px] justify-end flex-none">
            <span className="text-blue-400 font-bold tabular-nums">
              {metrics?.net_up_val || '0.0'}
            </span>
            <span className="opacity-20 text-[6.5px] font-light w-[15px]">
              {metrics?.net_up_unit?.[0] || 'K'}U
            </span>
          </div>

          <div className="w-[1px] h-2 bg-white/5 shrink-0" />

          {/* DOWN SLOT */}
          <div className="flex items-center gap-0.5 w-[50px] justify-end flex-none">
            <span className="text-blue-400 font-bold tabular-nums">
              {metrics?.net_down_val || '0.0'}
            </span>
            <span className="opacity-20 text-[6.5px] font-light w-[15px]">
              {metrics?.net_down_unit?.[0] || 'K'}D
            </span>
          </div>
        </div>

        {/* MODE & LINK (Fixed width to handle STANDARD/LIVE) */}
        <div className="flex items-center border-l border-white/5 h-full">
          <div className="flex items-center gap-2 px-3">
            <span className="opacity-20 font-light text-[7px] w-[30px] shrink-0">MODE</span>
            <div className="w-[50px] flex-none">
              <span className="text-white/40 truncate">{activeMode || 'STD'}</span>
            </div>
          </div>
          <div className="flex items-center gap-2 px-3 border-l border-white/5 h-full">
            <span className="opacity-20 font-light text-[7px] w-[25px] shrink-0">LINK</span>
            <div className="w-[30px] flex-none">
              <span className="text-green-400/40 font-bold">LIVE</span>
            </div>
          </div>
        </div>
      </div>

      {/* Dynamic Spacer */}
      <div className="flex-1 min-w-[30px]" />

      {/* Right side: Task Stream (Floating) */}
      <div className="w-[25%] min-w-[150px] flex items-center gap-3 overflow-hidden h-full border-l border-white/5 pl-4 flex-none">
        <span className="opacity-10 text-[7px] tracking-[0.1em] shrink-0 font-light italic">
          STREAM
        </span>
        <div className="flex-1 overflow-hidden relative h-full flex items-center">
          <div className="marquee-wrapper w-full">
            <div className="marquee-content text-white/[0.04] font-light tracking-normal normal-case italic py-1">
              {taskText}
              <span className="mx-20">&nbsp;</span>
              {taskText}
              <span className="mx-20">&nbsp;</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentStatus;
