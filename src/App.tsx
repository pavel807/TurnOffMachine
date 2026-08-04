import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Power, 
  Minus, 
  X, 
  ChevronRight, 
  Timer as TimerIcon, 
  RotateCcw, 
  Moon, 
  HardDrive,
  AlertCircle,
  Sun,
  CloudMoon
} from 'lucide-react';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { invoke } from '@tauri-apps/api/core';

// Импорты фонов
import DaySvg from './assets/day.svg';
import NightSvg from './assets/night.svg';

// --- Types ---
type ActionType = 'Shutdown' | 'Restart' | 'Sleep' | 'Hibernate';
type AppMode = 'loading' | 'idle' | 'counting';
type TimeOfDay = 'day' | 'night';

const ACTION_CONFIG: Record<ActionType, { icon: React.ElementType; label: string }> = {
  Shutdown: { icon: Power, label: 'Shutdown' },
  Restart: { icon: RotateCcw, label: 'Restart' },
  Sleep: { icon: Moon, label: 'Sleep' },
  Hibernate: { icon: HardDrive, label: 'Hibernate' },
};

// --- Loading Screen ---
const LoadingScreen = ({ onComplete, timeOfDay }: { onComplete: () => void; timeOfDay: TimeOfDay }) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(onComplete, 500);
          return 100;
        }
        return prev + Math.floor(Math.random() * 5) + 2;
      });
    }, 50);
    return () => clearInterval(interval);
  }, [onComplete]);

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning!';
    if (hour < 18) return 'Good afternoon!';
    return 'Good evening!';
  };

  const radius = 120;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (Math.min(progress, 100) / 100) * circumference;
  const accent = timeOfDay === 'day' ? '#f5a623' : '#a78bfa';

  return (
    <motion.div 
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 1.05 }}
      transition={{ duration: 0.6 }}
      className="absolute inset-0 z-50 flex items-center justify-center rounded-[2.5rem]"
      style={{ backgroundColor: 'rgba(30, 33, 40, 0.95)', backdropFilter: 'blur(20px)' }}
    >
      <div className="relative w-64 h-64 flex items-center justify-center">
        <svg className="absolute inset-0 w-full h-full -rotate-90">
          <circle cx="128" cy="128" r={radius} fill="rgba(42, 45, 53, 0.8)" stroke="rgba(255,255,255,0.08)" strokeWidth="8"/>
          <motion.circle
            cx="128" cy="128" r={radius} fill="none"
            stroke={accent}
            strokeWidth="8" strokeLinecap="round"
            strokeDasharray={circumference}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 0.15, ease: "easeOut" }}
          />
        </svg>
        <div className="relative z-10 flex flex-col items-center text-white">
          <Power size={36} strokeWidth={2} className="mb-3 drop-shadow-lg" style={{ color: accent }} />
          <span className="text-5xl font-bold tabular-nums tracking-tighter">{Math.min(progress, 100)}%</span>
          <span className="text-sm font-semibold mt-2 opacity-80">{getGreeting()}</span>
          <span className="text-xs opacity-50 mt-1 animate-pulse">loading...</span>
        </div>
      </div>
    </motion.div>
  );
};

// --- TitleBar ---
const TitleBar = ({ timeOfDay, onToggleTheme }: { timeOfDay: TimeOfDay; onToggleTheme: () => void }) => {
  const appWindow = getCurrentWindow();

  const handleDragStart = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button')) return;
    appWindow.startDragging();
  };

  return (
    <div 
      className="absolute top-0 left-0 w-full h-11 flex items-center justify-between px-4 select-none z-30 rounded-t-[2.5rem] cursor-grab active:cursor-grabbing"
      style={{ background: 'linear-gradient(to bottom, rgba(0,0,0,0.3), transparent)' }}
      onMouseDown={handleDragStart}
    >
      <div className="flex items-center gap-2 text-white pointer-events-none">
        <Power size={15} strokeWidth={2.5} className="text-white/90 drop-shadow-md" />
        <span className="text-[11px] font-bold tracking-wide uppercase text-white/90 drop-shadow-md">Turn Off Machine</span>
      </div>
      <div className="flex items-center gap-1 relative">
        <button 
          onMouseDown={(e) => e.stopPropagation()}
          onClick={onToggleTheme}
          className="w-6 h-6 flex items-center justify-center text-white/70 hover:text-white hover:bg-white/10 rounded-full transition-all border border-white/10"
          title="Toggle Day/Night"
        >
          {timeOfDay === 'day' ? <Sun size={12} strokeWidth={2.5} /> : <CloudMoon size={12} strokeWidth={2.5} />}
        </button>
        <button 
          onMouseDown={(e) => e.stopPropagation()}
          onClick={() => appWindow.minimize()}
          className="w-6 h-6 flex items-center justify-center text-white/70 hover:text-white hover:bg-white/10 rounded-full transition-all border border-white/10"
        >
          <Minus size={12} strokeWidth={3} />
        </button>
        <button 
          onMouseDown={(e) => e.stopPropagation()}
          onClick={() => appWindow.close()}
          className="w-6 h-6 flex items-center justify-center text-white/70 hover:text-red-400 hover:bg-red-500/20 rounded-full transition-all border border-white/10"
        >
          <X size={12} strokeWidth={3} />
        </button>
      </div>
    </div>
  );
};

// --- Clock ---
interface ClockProps { mode: AppMode; timeLeft: number; }

const Clock: React.FC<ClockProps> = ({ mode, timeLeft }) => {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    if (mode === 'idle') {
      const t = setInterval(() => setCurrentTime(new Date()), 1000);
      return () => clearInterval(t);
    }
  }, [mode]);

  let displayStr: { h: string; m: string; s: string };
  let ampm = '';

  if (mode === 'counting') {
    const h = Math.floor(timeLeft / 3600);
    const m = Math.floor((timeLeft % 3600) / 60);
    const s = timeLeft % 60;
    displayStr = { h: h.toString().padStart(2, '0'), m: m.toString().padStart(2, '0'), s: s.toString().padStart(2, '0') };
  } else {
    const hours = currentTime.getHours();
    const h12 = hours % 12 || 12;
    ampm = hours >= 12 ? 'PM' : 'AM';
    displayStr = { h: h12.toString().padStart(2, '0'), m: currentTime.getMinutes().toString().padStart(2, '0'), s: currentTime.getSeconds().toString().padStart(2, '0') };
  }

  return (
    <div className="flex items-baseline justify-center text-white font-light select-none mt-1 mb-1 leading-none drop-shadow-[0_4px_12px_rgba(0,0,0,0.5)]">
      <span className="text-[76px] tabular-nums tracking-tight">{displayStr.h}</span>
      <span className="text-[76px] tabular-nums tracking-tight mx-0.5 opacity-70">:</span>
      <span className="text-[76px] tabular-nums tracking-tight">{displayStr.m}</span>
      <div className="flex flex-col ml-2 self-end mb-2">
        <span className="text-2xl font-normal tabular-nums opacity-90 leading-none">:{displayStr.s}s</span>
        {mode === 'idle' && <span className="text-lg font-medium opacity-70 leading-none mt-0.5">{ampm}</span>}
      </div>
    </div>
  );
};

// --- TimerBox ---
interface TimerBoxProps { label: string; value: number; max: number; onChange: (v: number) => void; disabled: boolean; }

const TimerBox: React.FC<TimerBoxProps> = ({ label, value, max, onChange, disabled }) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    let v = parseInt(e.target.value) || 0;
    if (v < 0) v = 0;
    if (v > max) v = max;
    onChange(v);
  };

  return (
    <div className={`flex flex-col items-center justify-center rounded-2xl px-2 py-2 w-18 border border-white/10 transition-all duration-300 ${disabled ? 'opacity-40 pointer-events-none grayscale' : 'hover:border-white/20'}`}
         style={{ backgroundColor: 'rgba(42, 45, 53, 0.8)', backdropFilter: 'blur(12px)', boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.3)' }}>
      <div className="w-full rounded-lg py-1 mb-1 border border-black/40"
           style={{ backgroundColor: '#15171e', boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.8)' }}>
        <input type="number" value={value.toString().padStart(2, '0')} onChange={handleChange} disabled={disabled}
          className="w-full bg-transparent text-center text-lg font-semibold text-white focus:outline-none no-spinners appearance-none" />
      </div>
      <span className="text-[8px] text-white/50 font-bold tracking-widest uppercase">{label}</span>
    </div>
  );
};

// --- Toast ---
interface ToastProps { message: string; type: 'error' | 'success'; onClose: () => void; }

const Toast: React.FC<ToastProps> = ({ message, type, onClose }) => {
  useEffect(() => { const t = setTimeout(onClose, 4000); return () => clearTimeout(t); }, [onClose]);
  return (
    <motion.div initial={{ opacity: 0, y: 30, scale: 0.9 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 20, scale: 0.9 }}
      className={`fixed bottom-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 px-4 py-2 rounded-full border shadow-2xl ${type === 'error' ? 'border-red-500/30 text-red-200' : 'border-green-500/30 text-green-200'}`}
      style={{ backgroundColor: type === 'error' ? 'rgba(239,68,68,0.2)' : 'rgba(34,197,94,0.2)', backdropFilter: 'blur(20px)' }}>
      <AlertCircle size={14} /><span className="text-xs font-medium">{message}</span>
    </motion.div>
  );
};

// --- Main App ---
export default function App() {
  const [mode, setMode] = useState<AppMode>('loading');
  const [action, setAction] = useState<ActionType>('Shutdown');
  const [isOpen, setIsOpen] = useState(false);
  const [timer, setTimer] = useState({ h: 0, m: 0, s: 0 });
  const [timeLeft, setTimeLeft] = useState(0);
  const [toast, setToast] = useState<{ message: string; type: 'error' | 'success' } | null>(null);
  const [currentDate, setCurrentDate] = useState('');
  const [manualOverride, setManualOverride] = useState<TimeOfDay | null>(null);
  const dropRef = useRef<HTMLDivElement>(null);

  // Авто-определение времени суток
  const autoTimeOfDay: TimeOfDay = useMemo(() => {
    const hour = new Date().getHours();
    return hour >= 6 && hour < 18 ? 'day' : 'night';
  }, []);

  const timeOfDay: TimeOfDay = manualOverride ?? autoTimeOfDay;

  const toggleTheme = () => {
    setManualOverride(prev => {
      if (prev === 'day') return 'night';
      if (prev === 'night') return 'day';
      return autoTimeOfDay === 'day' ? 'night' : 'day';
    });
  };

  useEffect(() => {
    const now = new Date();
    const options: Intl.DateTimeFormatOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    setCurrentDate(now.toLocaleDateString('en-US', options));
  }, []);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropRef.current && !dropRef.current.contains(e.target as Node)) setIsOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (mode === 'counting' && timeLeft > 0) {
      interval = setInterval(() => setTimeLeft((prev) => prev - 1), 1000);
    } else if (mode === 'counting' && timeLeft === 0) {
      executeAction();
    }
    return () => clearInterval(interval);
  }, [mode, timeLeft, action]);

  const executeAction = useCallback(async () => {
    try {
      await invoke('execute_power_action', { action, delaySeconds: 0 });
      await getCurrentWindow().close();
    } catch (error) {
      console.error('Power action failed:', error);
      setToast({ message: `Permission denied. Check OS settings.`, type: 'error' });
      setMode('idle');
    }
  }, [action]);

  const handleStart = async () => {
    const totalSeconds = timer.h * 3600 + timer.m * 60 + timer.s;
    if (totalSeconds === 0) {
      await executeAction();
    } else {
      setTimeLeft(totalSeconds);
      setMode('counting');
      setToast({ message: `Timer started. ${action} in ${formatTime(totalSeconds)}.`, type: 'success' });
    }
  };

  const handleCancel = async () => {
    try { await invoke('cancel_power_action'); } catch (e) { console.warn(e); }
    setMode('idle');
    setTimeLeft(0);
    setToast({ message: 'Timer cancelled.', type: 'success' });
  };

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h > 0 ? h + 'h ' : ''}${m > 0 ? m + 'm ' : ''}${s}s`;
  };

  const CurrentIcon = ACTION_CONFIG[action].icon;
  const accentColor = timeOfDay === 'day' ? '#f5a623' : '#a78bfa';
  const bgColor = timeOfDay === 'day' ? '#2b241a' : '#0f0a1e';

  return (
    <div className="w-screen h-screen flex items-center justify-center" style={{ background: 'transparent' }}>
      <div className="relative w-112.5 h-187.5 rounded-[2.5rem] overflow-hidden border border-white/10"
           style={{ backgroundColor: bgColor, boxShadow: '0 25px 60px -10px rgba(0,0,0,0.5)', zoom: 0.8333 }}>
        
        {/* Loading */}
        <AnimatePresence>
          {mode === 'loading' && <LoadingScreen onComplete={() => setMode('idle')} timeOfDay={timeOfDay} />}
        </AnimatePresence>

        {/* Background — используем img для обоих SVG, они растянутся на всю ширину */}
        <div className="absolute -top-5 left-0 w-full h-[55%] z-0 pointer-events-none overflow-hidden rounded-t-[2.5rem]">
          <AnimatePresence mode="wait">
            <motion.img
              key={timeOfDay}
              src={timeOfDay === 'day' ? DaySvg : NightSvg}
              alt=""
              className="w-full h-full object-cover"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.8 }}
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
          </AnimatePresence>
          <div className="absolute bottom-0 left-0 w-full h-32" 
               style={{ background: `linear-gradient(to top, ${bgColor}, transparent)` }} />
        </div>

        <TitleBar timeOfDay={timeOfDay} onToggleTheme={toggleTheme} />

        <div className="relative z-10 flex flex-col items-center justify-between py-11 px-5 h-full">
          
          {/* Date & Clock */}
          <div className="flex flex-col items-center w-full mt-5">
            <p className="text-[11px] text-white/70 font-medium tracking-wide drop-shadow-md mb-1 capitalize">{currentDate}</p>
            <Clock mode={mode} timeLeft={timeLeft} />
          </div>

          {/* Power & Action */}
          <div className="flex flex-col items-center w-full -mt-1">
            <motion.div
              whileHover={mode === 'idle' ? { scale: 1.15, filter: `drop-shadow(0 0 15px ${accentColor})` } : {}}
              whileTap={mode === 'idle' ? { scale: 0.9 } : {}}
              onClick={mode === 'counting' ? handleCancel : undefined}
              className={`cursor-pointer mb-2 p-3 rounded-full transition-all duration-300 ${mode === 'counting' ? 'bg-red-500/20 hover:bg-red-500/30 animate-pulse' : 'hover:bg-white/5'}`}
            >
              <Power size={34} strokeWidth={1.8} className={`drop-shadow-lg transition-colors duration-300 ${mode === 'counting' ? 'text-red-400' : 'text-white'}`} />
            </motion.div>
            
            <p className="text-[11px] text-white/70 font-bold tracking-wide drop-shadow-md mb-3 h-4">
              {mode === 'idle' ? 'select an action below:' : `executing in ${formatTime(timeLeft)}...`}
            </p>

            <div className="w-full h-px bg-white/10 mb-3"></div>

            {/* Action Selector */}
            <div className="w-full flex items-center justify-center gap-2 mb-4 relative" ref={dropRef}>
              <span className="text-[9px] text-white/50 font-bold tracking-widest uppercase">action:</span>
              <div className="relative">
                <button
                  onClick={() => mode === 'idle' && setIsOpen(!isOpen)}
                  disabled={mode === 'counting'}
                  className={`flex items-center gap-2 pl-3 pr-1.5 py-1.5 rounded-full transition-all border border-white/10 ${mode === 'idle' ? 'hover:border-white/20 cursor-pointer' : 'opacity-50 cursor-not-allowed grayscale'}`}
                  style={{ backgroundColor: 'rgba(42, 45, 53, 0.9)', backdropFilter: 'blur(12px)' }}
                >
                  <CurrentIcon size={13} style={{ color: accentColor }} />
                  <span className="text-[11px] font-bold text-white pr-1">{ACTION_CONFIG[action].label}</span>
                  <div className="w-4 h-4 flex items-center justify-center bg-white/5 rounded-full">
                    <ChevronRight size={9} className={`text-white/50 transition-transform duration-200 ${isOpen ? 'rotate-90' : ''}`} />
                  </div>
                </button>

                <AnimatePresence>
                  {isOpen && mode === 'idle' && (
                    <motion.div initial={{ opacity: 0, y: -5, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -5, scale: 0.95 }}
                      className="absolute top-full left-0 right-0 mt-2 rounded-xl overflow-hidden shadow-2xl z-50 min-w-32.5 border border-white/10"
                      style={{ backgroundColor: 'rgba(37, 40, 48, 0.95)', backdropFilter: 'blur(20px)' }}>
                      {(Object.keys(ACTION_CONFIG) as ActionType[]).map((key) => {
                        const Icon = ACTION_CONFIG[key].icon;
                        const isActive = action === key;
                        return (
                          <button key={key} onClick={() => { setAction(key); setIsOpen(false); }}
                            className="w-full flex items-center gap-2 px-3 py-2 text-[11px] transition-colors"
                            style={isActive 
                              ? { backgroundColor: `${accentColor}20`, color: accentColor } 
                              : { color: 'rgba(255,255,255,0.7)' }}>
                            <Icon size={13} /><span className="font-bold">{ACTION_CONFIG[key].label}</span>
                          </button>
                        );
                      })}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* Timer */}
            <div className="flex gap-2.5 w-full justify-center mb-5">
              <TimerBox label="hours" value={timer.h} max={99} onChange={(v) => setTimer(p => ({ ...p, h: v }))} disabled={mode === 'counting'} />
              <TimerBox label="minutes" value={timer.m} max={59} onChange={(v) => setTimer(p => ({ ...p, m: v }))} disabled={mode === 'counting'} />
              <TimerBox label="seconds" value={timer.s} max={59} onChange={(v) => setTimer(p => ({ ...p, s: v }))} disabled={mode === 'counting'} />
            </div>
          </div>

          {/* Start/Cancel */}
          <div className="w-full flex justify-center pb-1">
            <AnimatePresence mode="wait">
              {mode === 'idle' ? (
                <motion.button key="start" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
                  whileHover={{ scale: 1.05, boxShadow: `0 0 30px ${accentColor}80` }} whileTap={{ scale: 0.95 }} onClick={handleStart}
                  className="flex items-center justify-center gap-2 px-10 py-3 rounded-full font-extrabold text-[11px] tracking-widest uppercase relative overflow-hidden group"
                  style={{ 
                    background: timeOfDay === 'day' ? 'linear-gradient(to right, #f5a623, #f7c948)' : 'linear-gradient(to right, #7c3aed, #a78bfa)', 
                    boxShadow: `0 6px 20px ${accentColor}60`, 
                    color: timeOfDay === 'day' ? '#1a1a1a' : '#fff' 
                  }}>
                  <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-out" />
                  <span className="relative z-10">Start Now</span>
                  <TimerIcon size={15} className="relative z-10 opacity-80" strokeWidth={2.5} />
                </motion.button>
              ) : (
                <motion.button key="cancel" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
                  whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={handleCancel}
                  className="flex items-center justify-center gap-2 px-10 py-3 rounded-full border border-red-500/50 text-red-300 font-extrabold text-[11px] tracking-widest uppercase transition-colors"
                  style={{ backgroundColor: 'rgba(239, 68, 68, 0.2)', backdropFilter: 'blur(12px)' }}>
                  <X size={15} strokeWidth={2.5} /><span>Cancel Timer</span>
                </motion.button>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Toast */}
        <AnimatePresence>
          {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
        </AnimatePresence>
      </div>
    </div>
  );
}