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
  CloudMoon,
  Settings,
  Check,
  Download,
  Loader2
} from 'lucide-react';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { invoke } from '@tauri-apps/api/core';
import { getVersion } from '@tauri-apps/api/app';
import { enable, disable, isEnabled } from '@tauri-apps/plugin-autostart';
import { check, type Update } from '@tauri-apps/plugin-updater';
import { relaunch } from '@tauri-apps/plugin-process';
import { openUrl } from '@tauri-apps/plugin-opener';

// Импорты фонов
import DaySvg from './assets/day.svg';
import NightSvg from './assets/night.svg';

// --- Types ---
type ActionType = 'Shutdown' | 'Restart' | 'Sleep' | 'Hibernate';
type AppMode = 'loading' | 'idle' | 'counting' | 'confirming';
type TimeOfDay = 'day' | 'night';
type Lang = 'en' | 'ru';

const ACTION_CONFIG: Record<ActionType, { icon: React.ElementType; label: Record<Lang, string>; verb: Record<Lang, string> }> = {
  Shutdown: { icon: Power, label: { en: 'Shutdown', ru: 'Выключение' }, verb: { en: 'shutdown', ru: 'выключить' } },
  Restart: { icon: RotateCcw, label: { en: 'Restart', ru: 'Перезагрузка' }, verb: { en: 'restart', ru: 'перезагрузить' } },
  Sleep: { icon: Moon, label: { en: 'Sleep', ru: 'Сон' }, verb: { en: 'sleep', ru: 'перевести в сон' } },
  Hibernate: { icon: HardDrive, label: { en: 'Hibernate', ru: 'Гибернация' }, verb: { en: 'hibernate', ru: 'перевести в гибернацию' } },
};

// --- I18n ---
interface I18n {
  greetingMorning: string;
  greetingAfternoon: string;
  greetingEvening: string;
  loading: string;
  toggleTheme: string;
  selectAction: string;
  executingIn: (time: string) => string;
  actionLabel: string;
  hours: string;
  minutes: string;
  seconds: string;
  startNow: string;
  cancelTimer: string;
  timerStarted: (action: string, time: string) => string;
  timerCancelled: string;
  actionCancelled: string;
  permissionDenied: string;
  settings: string;
  launchAtLogin: string;
  launchAtLoginSub: string;
  alwaysOnTop: string;
  alwaysOnTopSub: string;
  confirmBeforeAction: string;
  confirmBeforeActionSub: string;
  rememberLastAction: string;
  rememberLastActionSub: string;
  soundAtEnd: string;
  soundAtEndSub: string;
  language: string;
  confirmTitle: (action: string) => string;
  confirmDesc: (action: string) => string;
  cancel: string;
  yes: string;
  autostartEnabled: string;
  autostartDisabled: string;
  autostartFailed: string;
  checkUpdates: string;
  checkingUpdates: string;
  upToDate: string;
  checkFailed: string;
  updateAvailable: (v: string) => string;
  updateDesc: (v: string) => string;
  downloadAndInstall: string;
  later: string;
  downloading: (p: number) => string;
  installing: string;
  openReleases: string;
  updateError: string;
}

const T: Record<Lang, I18n> = {
  en: {
    greetingMorning: 'Good morning!',
    greetingAfternoon: 'Good afternoon!',
    greetingEvening: 'Good evening!',
    loading: 'loading...',
    toggleTheme: 'Toggle Day/Night',
    selectAction: 'select an action below:',
    executingIn: (time) => `executing in ${time}...`,
    actionLabel: 'action:',
    hours: 'hours',
    minutes: 'minutes',
    seconds: 'seconds',
    startNow: 'Start Now',
    cancelTimer: 'Cancel Timer',
    timerStarted: (action, time) => `Timer started. ${action} in ${time}.`,
    timerCancelled: 'Timer cancelled.',
    actionCancelled: 'Action cancelled.',
    permissionDenied: 'Permission denied. Check OS settings.',
    settings: 'Settings',
    launchAtLogin: 'Launch at login',
    launchAtLoginSub: 'Start automatically with your system',
    alwaysOnTop: 'Always on top',
    alwaysOnTopSub: 'Keep the window above others',
    confirmBeforeAction: 'Confirm before action',
    confirmBeforeActionSub: 'Ask before performing the action',
    rememberLastAction: 'Remember last action',
    rememberLastActionSub: 'Restore your last choice on launch',
    soundAtEnd: 'Sound at end',
    soundAtEndSub: 'Play a sound when the timer ends',
    language: 'Language',
    confirmTitle: (action) => `${action} now?`,
    confirmDesc: (action) => `The timer has finished. Do you want to ${action} your computer now?`,
    cancel: 'Cancel',
    yes: 'Yes',
    autostartEnabled: 'Launch at login enabled.',
    autostartDisabled: 'Launch at login disabled.',
    autostartFailed: 'Failed to change autostart. Run the built app.',
    checkUpdates: 'Check for updates',
    checkingUpdates: 'Checking...',
    upToDate: 'You are up to date.',
    checkFailed: 'Failed to check for updates.',
    updateAvailable: (v) => `Update available: v${v}`,
    updateDesc: (v) => `A new version (v${v}) is available. Download and install it?`,
    downloadAndInstall: 'Download & Install',
    later: 'Later',
    downloading: (p) => `Downloading... ${p}%`,
    installing: 'Installing...',
    openReleases: 'Open Releases page',
    updateError: 'Update failed. Download manually instead.',
  },
  ru: {
    greetingMorning: 'Доброе утро!',
    greetingAfternoon: 'Добрый день!',
    greetingEvening: 'Добрый вечер!',
    loading: 'загрузка...',
    toggleTheme: 'День/Ночь',
    selectAction: 'выберите действие ниже:',
    executingIn: (time) => `выполнение через ${time}...`,
    actionLabel: 'действие:',
    hours: 'часы',
    minutes: 'минуты',
    seconds: 'секунды',
    startNow: 'Старт',
    cancelTimer: 'Отмена',
    timerStarted: (action, time) => `Таймер запущен. ${action} через ${time}.`,
    timerCancelled: 'Таймер отменён.',
    actionCancelled: 'Действие отменено.',
    permissionDenied: 'Доступ запрещён. Проверьте настройки ОС.',
    settings: 'Настройки',
    launchAtLogin: 'Запуск при входе',
    launchAtLoginSub: 'Автоматический запуск с системой',
    alwaysOnTop: 'Поверх всех окон',
    alwaysOnTopSub: 'Окно всегда поверх других окон',
    confirmBeforeAction: 'Подтверждать действие',
    confirmBeforeActionSub: 'Спрашивать перед выполнением действия',
    rememberLastAction: 'Запоминать действие',
    rememberLastActionSub: 'Восстанавливать выбор при запуске',
    soundAtEnd: 'Звук по окончании',
    soundAtEndSub: 'Сигнал, когда таймер закончится',
    language: 'Язык',
    confirmTitle: (action) => `${action.charAt(0).toUpperCase()}${action.slice(1)} сейчас?`,
    confirmDesc: (action) => `Таймер завершён. ${action.charAt(0).toUpperCase()}${action.slice(1)} компьютер сейчас?`,
    cancel: 'Отмена',
    yes: 'Да',
    autostartEnabled: 'Автозапуск включён.',
    autostartDisabled: 'Автозапуск выключен.',
    autostartFailed: 'Не удалось изменить автозапуск. Запустите собранное приложение.',
    checkUpdates: 'Проверить обновления',
    checkingUpdates: 'Проверка...',
    upToDate: 'Установлена последняя версия.',
    checkFailed: 'Не удалось проверить обновления.',
    updateAvailable: (v) => `Доступно обновление: v${v}`,
    updateDesc: (v) => `Доступна новая версия (v${v}). Скачать и установить?`,
    downloadAndInstall: 'Скачать и установить',
    later: 'Позже',
    downloading: (p) => `Загрузка... ${p}%`,
    installing: 'Установка...',
    openReleases: 'Открыть страницу релизов',
    updateError: 'Ошибка обновления. Скачайте вручную.',
  },
};

// --- Loading Screen ---
const LoadingScreen = ({ onComplete, timeOfDay, lang }: { onComplete: () => void; timeOfDay: TimeOfDay; lang: Lang }) => {
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

  const t = T[lang];
  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return t.greetingMorning;
    if (hour < 18) return t.greetingAfternoon;
    return t.greetingEvening;
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
    >
      <div className="relative flex items-center justify-center">
        <div className="absolute rounded-full border border-white/10"
          style={{ width: 288, height: 288, backgroundColor: 'rgba(24, 26, 32, 0.9)', backdropFilter: 'blur(20px)', boxShadow: '0 20px 50px -10px rgba(0,0,0,0.6)' }} />
        <div className="relative w-64 h-64 flex items-center justify-center">
          <svg className="absolute inset-0 w-full h-full -rotate-90">
            <motion.circle
              cx="128" cy="128" r={radius} fill="none"
              stroke={accent}
              strokeWidth="8" strokeLinecap="round"
              strokeDasharray={circumference}
              animate={{ strokeDashoffset: offset }}
              transition={{ duration: 0.15, ease: "easeOut" }}
            />
          </svg>
          <div className="relative z-10 flex flex-col items-center text-white drop-shadow-[0_2px_8px_rgba(0,0,0,0.9)]">
            <Power size={36} strokeWidth={2} className="mb-3" style={{ color: accent }} />
            <span className="text-5xl font-bold tabular-nums tracking-tighter">{Math.min(progress, 100)}%</span>
            <span className="text-sm font-semibold mt-2 opacity-90">{getGreeting()}</span>
            <span className="text-xs opacity-60 mt-1 animate-pulse">{t.loading}</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// --- TitleBar ---
interface TitleBarProps {
  timeOfDay: TimeOfDay;
  onToggleTheme: () => void;
  onToggleSettings: () => void;
  settingsOpen: boolean;
  t: I18n;
}

const TitleBar: React.FC<TitleBarProps> = ({ timeOfDay, onToggleTheme, onToggleSettings, settingsOpen, t }) => {
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
          onClick={onToggleSettings}
          className={`w-6 h-6 flex items-center justify-center rounded-full transition-all border border-white/10 ${settingsOpen ? 'bg-white/15 text-white' : 'text-white/70 hover:text-white hover:bg-white/10'}`}
          title={t.settings}
        >
          <Settings size={12} strokeWidth={2.5} />
        </button>
        <button 
          onMouseDown={(e) => e.stopPropagation()}
          onClick={onToggleTheme}
          className="w-6 h-6 flex items-center justify-center text-white/70 hover:text-white hover:bg-white/10 rounded-full transition-all border border-white/10"
          title={t.toggleTheme}
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
  const onCloseRef = useRef(onClose);
  useEffect(() => { onCloseRef.current = onClose; });
  useEffect(() => {
    const t = setTimeout(() => onCloseRef.current(), 4000);
    return () => clearTimeout(t);
  }, []);
  return (
    <motion.div initial={{ opacity: 0, y: 30, scale: 0.9 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 20, scale: 0.9 }}
      className={`fixed bottom-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 pl-4 pr-1.5 py-1.5 rounded-full border shadow-2xl ${type === 'error' ? 'border-red-500/30 text-red-200' : 'border-green-500/30 text-green-200'}`}
      style={{ backgroundColor: type === 'error' ? 'rgba(239,68,68,0.2)' : 'rgba(34,197,94,0.2)', backdropFilter: 'blur(20px)' }}>
      <AlertCircle size={14} /><span className="text-xs font-medium">{message}</span>
      <button onClick={onClose} className="ml-1 p-1 rounded-full hover:bg-white/10 transition-colors">
        <X size={11} strokeWidth={3} />
      </button>
    </motion.div>
  );
};

// --- SettingsPanel ---
type BehaviorSettingKey = 'alwaysOnTop' | 'confirmBeforeAction' | 'rememberAction' | 'soundAtEnd';

interface BehaviorSettings {
  alwaysOnTop: boolean;
  confirmBeforeAction: boolean;
  rememberAction: boolean;
  soundAtEnd: boolean;
}

const DEFAULT_BEHAVIOR_SETTINGS: BehaviorSettings = {
  alwaysOnTop: false,
  confirmBeforeAction: false,
  rememberAction: false,
  soundAtEnd: true,
};

const Switch: React.FC<{ on: boolean; accent: string; onClick: () => void }> = ({ on, accent, onClick }) => (
  <button
    onClick={onClick}
    className={`relative shrink-0 w-9 h-5 rounded-full transition-all duration-200 ${on ? '' : 'bg-white/15'}`}
    style={on ? { backgroundColor: accent, boxShadow: `0 0 10px ${accent}66` } : {}}
  >
    <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-200 ${on ? 'left-[18px]' : 'left-0.5'}`} />
  </button>
);

const SettingRow: React.FC<{ title: string; sub: string; on: boolean; accent: string; onToggle: () => void }> = ({ title, sub, on, accent, onToggle }) => (
  <div className="flex items-center justify-between gap-3 py-2">
    <div className="flex flex-col gap-0.5">
      <span className="text-[11px] font-semibold text-white leading-none">{title}</span>
      <span className="text-[9px] text-white/50 leading-tight mt-0.5">{sub}</span>
    </div>
    <Switch on={on} accent={accent} onClick={onToggle} />
  </div>
);

const SettingButtonRow: React.FC<{ title: string; sub: string; busyText: string; busy: boolean; accent: string; onClick: () => void }> = ({ title, sub, busyText, busy, accent, onClick }) => (
  <button onClick={onClick} disabled={busy}
    className="w-full flex items-center justify-between gap-3 py-2 rounded-lg transition-colors disabled:opacity-60 hover:bg-white/5 px-1 -mx-1">
    <div className="flex flex-col gap-0.5 text-left">
      <span className="text-[11px] font-semibold text-white leading-none">{title}</span>
      <span className="text-[9px] text-white/50 leading-tight mt-0.5">{busy ? busyText : ''}</span>
    </div>
    {busy
      ? <Loader2 size={13} className="animate-spin text-white/60 shrink-0" />
      : <span className="text-[9px] font-bold tracking-widest uppercase px-2 py-1 rounded-full shrink-0" style={{ color: accent }}>{sub}</span>}
  </button>
);

interface SettingsPanelProps {
  autoStart: boolean;
  settings: BehaviorSettings;
  onToggleAutoStart: () => void;
  onToggleSetting: (key: BehaviorSettingKey) => void;
  accent: string;
  lang: Lang;
  onLangChange: (l: Lang) => void;
  t: I18n;
  appVersion: string;
  checkingUpdate: boolean;
  onCheckUpdates: () => void;
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({ autoStart, settings, onToggleAutoStart, onToggleSetting, accent, lang, onLangChange, t, appVersion, checkingUpdate, onCheckUpdates }) => (
  <motion.div
    initial={{ opacity: 0, y: -8, scale: 0.96 }}
    animate={{ opacity: 1, y: 0, scale: 1 }}
    exit={{ opacity: 0, y: -8, scale: 0.96 }}
    transition={{ duration: 0.15 }}
    className="w-64 rounded-2xl border border-white/10 shadow-2xl p-3.5 max-h-[calc(100vh-120px)] overflow-y-auto"
    style={{ backgroundColor: 'rgba(37,40,48,0.97)', backdropFilter: 'blur(20px)' }}
  >
    <p className="text-[9px] font-bold tracking-widest uppercase text-white/50 mb-1 flex items-center gap-1.5">
      <Settings size={10} /> {t.settings}
    </p>
    <SettingRow title={t.launchAtLogin} sub={t.launchAtLoginSub}
      on={autoStart} accent={accent} onToggle={onToggleAutoStart} />
    <div className="h-px bg-white/10 my-1" />
    <SettingRow title={t.alwaysOnTop} sub={t.alwaysOnTopSub}
      on={settings.alwaysOnTop} accent={accent} onToggle={() => onToggleSetting('alwaysOnTop')} />
    <SettingRow title={t.confirmBeforeAction} sub={t.confirmBeforeActionSub}
      on={settings.confirmBeforeAction} accent={accent} onToggle={() => onToggleSetting('confirmBeforeAction')} />
    <SettingRow title={t.rememberLastAction} sub={t.rememberLastActionSub}
      on={settings.rememberAction} accent={accent} onToggle={() => onToggleSetting('rememberAction')} />
    <SettingRow title={t.soundAtEnd} sub={t.soundAtEndSub}
      on={settings.soundAtEnd} accent={accent} onToggle={() => onToggleSetting('soundAtEnd')} />
    <div className="h-px bg-white/10 my-1" />
    <SettingButtonRow title={t.checkUpdates} sub={`v${appVersion}`} busyText={t.checkingUpdates}
      busy={checkingUpdate} accent={accent} onClick={onCheckUpdates} />
    <div className="h-px bg-white/10 my-1" />
    <div className="flex items-center justify-between gap-3 py-2">
      <span className="text-[11px] font-semibold text-white leading-none">{t.language}</span>
      <div className="flex rounded-full border border-white/10 overflow-hidden">
        {(['en', 'ru'] as Lang[]).map((l) => (
          <button key={l} onClick={() => onLangChange(l)}
            className={`px-2.5 py-1 text-[9px] font-bold tracking-widest uppercase transition-colors ${lang === l ? 'text-white' : 'text-white/50 hover:text-white/80'}`}
            style={lang === l ? { backgroundColor: accent } : {}}>
            {l === 'en' ? 'EN' : 'RU'}
          </button>
        ))}
      </div>
    </div>
  </motion.div>
);

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
  const [autoStart, setAutoStart] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<BehaviorSettings>(DEFAULT_BEHAVIOR_SETTINGS);
  const [confirming, setConfirming] = useState(false);
  const [lang, setLang] = useState<Lang>(() => {
    try { return (localStorage.getItem('lang') as Lang) || 'en'; } catch { return 'en'; }
  });
  const [appVersion, setAppVersion] = useState('');
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [updateModal, setUpdateModal] = useState<
    | { status: 'available'; version: string }
    | { status: 'downloading'; progress: number }
    | { status: 'installing' }
    | { status: 'error' }
    | null
  >(null);
  const updateRef = useRef<Update | null>(null);
  const autoChecked = useRef(false);
  const dropRef = useRef<HTMLDivElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);

  const t = T[lang];
  const actionLabel = ACTION_CONFIG[action].label[lang];
  const actionVerb = ACTION_CONFIG[action].verb[lang];

  const changeLang = (l: Lang) => {
    setLang(l);
    try { localStorage.setItem('lang', l); } catch (e) { console.warn(e); }
  };

  useEffect(() => {
    getVersion().then(setAppVersion).catch(() => {});
  }, []);

  const checkForUpdates = useCallback(async (manual: boolean) => {
    setCheckingUpdate(true);
    try {
      const update = await check();
      if (update) {
        updateRef.current = update;
        setUpdateModal({ status: 'available', version: update.version });
      } else if (manual) {
        setToast({ message: T[lang].upToDate, type: 'success' });
      }
    } catch (e) {
      console.error('Update check failed:', e);
      if (manual) setToast({ message: T[lang].checkFailed, type: 'error' });
    } finally {
      setCheckingUpdate(false);
    }
  }, [lang]);

  useEffect(() => {
    if (mode === 'idle' && !autoChecked.current) {
      autoChecked.current = true;
      setTimeout(() => checkForUpdates(false), 3000);
    }
  }, [mode, checkForUpdates]);

  const handleDownloadUpdate = async () => {
    const update = updateRef.current;
    if (!update) return;
    try {
      setUpdateModal({ status: 'downloading', progress: 0 });
      let downloaded = 0;
      let total = 0;
      await update.download((event) => {
        if (event.event === 'Started') {
          total = event.data.contentLength ?? 0;
        } else if (event.event === 'Progress') {
          downloaded += event.data.chunkLength;
          const pct = total > 0 ? Math.min(100, Math.round((downloaded / total) * 100)) : 0;
          setUpdateModal({ status: 'downloading', progress: pct });
        }
      });
      setUpdateModal({ status: 'installing' });
      await update.install();
      await relaunch();
    } catch (e) {
      console.error('Update failed:', e);
      setUpdateModal({ status: 'error' });
    }
  };

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
      if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) setSettingsOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  useEffect(() => {
    isEnabled().then(setAutoStart).catch(() => {});
  }, []);

  useEffect(() => {
    try {
      const raw = localStorage.getItem('settings');
      if (raw) setSettings({ ...DEFAULT_BEHAVIOR_SETTINGS, ...JSON.parse(raw) });
    } catch (e) { console.warn('Failed to load settings:', e); }
  }, []);

  useEffect(() => {
    try {
      const saved = localStorage.getItem('selectedAction');
      if (saved && saved in ACTION_CONFIG) setAction(saved as ActionType);
    } catch (e) { console.warn('Failed to load action:', e); }
  }, []);

  useEffect(() => {
    getCurrentWindow().setAlwaysOnTop(settings.alwaysOnTop).catch(() => {});
  }, [settings.alwaysOnTop]);

  const updateSetting = (key: BehaviorSettingKey) => {
    setSettings(prev => {
      const next = { ...prev, [key]: !prev[key] };
      try { localStorage.setItem('settings', JSON.stringify(next)); } catch (e) { console.warn(e); }
      return next;
    });
  };

  const toggleAutoStart = async () => {
    try {
      if (autoStart) {
        await disable();
      } else {
        await enable();
      }
      setAutoStart(!autoStart);
      setToast({ message: !autoStart ? t.autostartEnabled : t.autostartDisabled, type: 'success' });
    } catch (e) {
      console.error('Autostart change failed:', e);
      setToast({ message: t.autostartFailed, type: 'error' });
    }
  };

  const playEndSound = useCallback(() => {
    if (!settings.soundAtEnd) return;
    try {
      const ctx = new AudioContext();
      [0, 0.35, 0.7].forEach((t, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = i === 2 ? 1046.5 : 784;
        osc.type = 'sine';
        gain.gain.setValueAtTime(0.0001, ctx.currentTime + t);
        gain.gain.exponentialRampToValueAtTime(0.4, ctx.currentTime + t + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + t + 0.28);
        osc.start(ctx.currentTime + t);
        osc.stop(ctx.currentTime + t + 0.32);
      });
      setTimeout(() => ctx.close(), 1500);
    } catch (e) { console.warn('Sound failed:', e); }
  }, [settings.soundAtEnd]);

  const executeAction = useCallback(async () => {
    try {
      await invoke('execute_power_action', { action, delaySeconds: 0 });
      await getCurrentWindow().close();
    } catch (error) {
      console.error('Power action failed:', error);
      setToast({ message: t.permissionDenied, type: 'error' });
      setMode('idle');
    }
  }, [action, t]);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (mode === 'counting' && timeLeft > 0) {
      interval = setInterval(() => setTimeLeft((prev) => prev - 1), 1000);
    } else if (mode === 'counting' && timeLeft === 0) {
      if (settings.confirmBeforeAction) {
        playEndSound();
        setMode('confirming');
        setConfirming(true);
      } else {
        playEndSound();
        executeAction();
      }
    }
    return () => clearInterval(interval);
  }, [mode, timeLeft, action, settings.confirmBeforeAction, playEndSound, executeAction]);

  const handleStart = async () => {
    const totalSeconds = timer.h * 3600 + timer.m * 60 + timer.s;
    if (totalSeconds === 0) {
      await executeAction();
    } else {
      setTimeLeft(totalSeconds);
      setMode('counting');
      setToast({ message: t.timerStarted(actionLabel, formatTime(totalSeconds)), type: 'success' });
    }
  };

  const handleCancel = async () => {
    try { await invoke('cancel_power_action'); } catch (e) { console.warn(e); }
    setMode('idle');
    setConfirming(false);
    setTimeLeft(0);
    setToast({ message: t.timerCancelled, type: 'success' });
  };

  const handleConfirmAction = async () => {
    setConfirming(false);
    await executeAction();
  };

  const handleDenyAction = () => {
    setConfirming(false);
    setMode('idle');
    setTimeLeft(0);
    setToast({ message: t.actionCancelled, type: 'success' });
  };

  const selectAction = (a: ActionType) => {
    setAction(a);
    setIsOpen(false);
    if (settings.rememberAction) {
      try { localStorage.setItem('selectedAction', a); } catch (e) { console.warn(e); }
    }
  };

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (lang === 'ru') {
      return `${h > 0 ? h + 'ч ' : ''}${m > 0 ? m + 'м ' : ''}${s}с`;
    }
    return `${h > 0 ? h + 'h ' : ''}${m > 0 ? m + 'm ' : ''}${s}s`;
  };

  const CurrentIcon = ACTION_CONFIG[action].icon;
  const accentColor = timeOfDay === 'day' ? '#f5a623' : '#a78bfa';
  const bgColor = timeOfDay === 'day' ? '#2b241a' : '#0f0a1e';

  return (
    <div className="w-screen h-screen flex items-center justify-center" style={{ background: 'transparent' }}>
      <div className={`relative w-112.5 h-187.5 rounded-[2.5rem] overflow-hidden ${mode === 'loading' ? 'border-transparent' : 'border border-white/10'}`}
           style={{ backgroundColor: mode === 'loading' ? 'transparent' : bgColor, zoom: 0.8333 }}>
        
        {/* Loading — пока идёт загрузка, интерфейс скрыт, окно прозрачное (виден рабочий стол) */}
        <AnimatePresence>
          {mode === 'loading' && <LoadingScreen onComplete={() => setMode('idle')} timeOfDay={timeOfDay} lang={lang} />}
        </AnimatePresence>

        {mode !== 'loading' && (<>
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

        <TitleBar timeOfDay={timeOfDay} onToggleTheme={toggleTheme} onToggleSettings={() => setSettingsOpen(o => !o)} settingsOpen={settingsOpen} t={t} />

        {/* Settings */}
        <div ref={settingsRef} className="absolute top-12 right-3 z-40 pointer-events-auto">
          <AnimatePresence>
            {settingsOpen && (
              <SettingsPanel autoStart={autoStart} settings={settings} onToggleAutoStart={toggleAutoStart} onToggleSetting={updateSetting} accent={accentColor} lang={lang} onLangChange={changeLang} t={t} appVersion={appVersion} checkingUpdate={checkingUpdate} onCheckUpdates={() => checkForUpdates(true)} />
            )}
          </AnimatePresence>
        </div>

        {/* Confirm before action */}
        <AnimatePresence>
          {confirming && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 z-40 flex items-center justify-center rounded-[2.5rem]"
              style={{ backgroundColor: 'rgba(10, 11, 14, 0.6)', backdropFilter: 'blur(8px)' }}
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.9, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: 10 }}
                transition={{ duration: 0.2 }}
                className="w-64 p-4 rounded-2xl border border-white/10 shadow-2xl"
                style={{ backgroundColor: 'rgba(37, 40, 48, 0.97)', backdropFilter: 'blur(20px)' }}
              >
                <div className="flex items-center gap-2.5 mb-2">
                  <CurrentIcon size={18} style={{ color: accentColor }} />
                  <span className="text-sm font-bold text-white">{t.confirmTitle(lang === 'en' ? actionLabel : actionVerb)}</span>
                </div>
                <p className="text-[10px] text-white/60 leading-snug mb-4">{t.confirmDesc(actionVerb)}</p>
                <div className="flex gap-2">
                  <button onClick={handleDenyAction}
                    className="flex-1 px-3 py-2 rounded-xl text-[10px] font-bold tracking-wide uppercase border border-white/15 text-white/80 hover:bg-white/10 transition-colors">
                    {t.cancel}
                  </button>
                  <button onClick={handleConfirmAction}
                    className="flex-1 px-3 py-2 rounded-xl text-[10px] font-bold tracking-wide uppercase transition-transform"
                    style={{ background: `linear-gradient(to right, ${accentColor}, ${accentColor}cc)`, color: timeOfDay === 'day' ? '#1a1a1a' : '#fff', boxShadow: `0 4px 15px ${accentColor}60` }}>
                    {t.yes}
                  </button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

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
              onClick={mode === 'counting' || mode === 'confirming' ? handleCancel : undefined}
              className={`cursor-pointer mb-2 p-3 rounded-full transition-all duration-300 ${mode === 'counting' || mode === 'confirming' ? 'bg-red-500/20 hover:bg-red-500/30 animate-pulse' : 'hover:bg-white/5'}`}
            >
              <Power size={34} strokeWidth={1.8} className={`drop-shadow-lg transition-colors duration-300 ${mode === 'counting' || mode === 'confirming' ? 'text-red-400' : 'text-white'}`} />
            </motion.div>
            
            <p className="text-[11px] text-white/70 font-bold tracking-wide drop-shadow-md mb-3 h-4">
              {mode === 'idle' ? t.selectAction : t.executingIn(formatTime(timeLeft))}
            </p>

            <div className="w-full h-px bg-white/10 mb-3"></div>

            {/* Action Selector */}
            <div className="w-full flex items-center justify-center gap-2 mb-4 relative" ref={dropRef}>
              <span className="text-[9px] text-white/50 font-bold tracking-widest uppercase">{t.actionLabel}</span>
              <div className="relative">
                <button
                  onClick={() => mode === 'idle' && setIsOpen(!isOpen)}
                  disabled={mode === 'counting'}
                  className={`flex items-center gap-2 pl-3 pr-1.5 py-1.5 rounded-full transition-all border border-white/10 ${mode === 'idle' ? 'hover:border-white/20 cursor-pointer' : 'opacity-50 cursor-not-allowed grayscale'}`}
                  style={{ backgroundColor: 'rgba(42, 45, 53, 0.9)', backdropFilter: 'blur(12px)' }}
                >
                  <CurrentIcon size={13} style={{ color: accentColor }} />
                  <span className="text-[11px] font-bold text-white pr-1">{actionLabel}</span>
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
                          <button key={key} onClick={() => selectAction(key)}
                            className="w-full flex items-center gap-2 px-3 py-2 text-[11px] transition-colors hover:bg-white/5"
                            style={isActive 
                              ? { backgroundColor: `${accentColor}20`, color: accentColor } 
                              : { color: 'rgba(255,255,255,0.7)' }}>
                            <Icon size={13} />
                            <span className="font-bold flex-1 text-left">{ACTION_CONFIG[key].label[lang]}</span>
                            {isActive && <Check size={12} strokeWidth={3} />}
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
              <TimerBox label={t.hours} value={timer.h} max={99} onChange={(v) => setTimer(p => ({ ...p, h: v }))} disabled={mode === 'counting'} />
              <TimerBox label={t.minutes} value={timer.m} max={59} onChange={(v) => setTimer(p => ({ ...p, m: v }))} disabled={mode === 'counting'} />
              <TimerBox label={t.seconds} value={timer.s} max={59} onChange={(v) => setTimer(p => ({ ...p, s: v }))} disabled={mode === 'counting'} />
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
                  <span className="relative z-10">{t.startNow}</span>
                  <TimerIcon size={15} className="relative z-10 opacity-80" strokeWidth={2.5} />
                </motion.button>
              ) : (
                <motion.button key="cancel" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
                  whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={handleCancel}
                  className="flex items-center justify-center gap-2 px-10 py-3 rounded-full border border-red-500/50 text-red-300 font-extrabold text-[11px] tracking-widest uppercase transition-colors"
                  style={{ backgroundColor: 'rgba(239, 68, 68, 0.2)', backdropFilter: 'blur(12px)' }}>
                  <X size={15} strokeWidth={2.5} /><span>{t.cancelTimer}</span>
                </motion.button>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Update modal */}
        <AnimatePresence>
          {updateModal && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 z-40 flex items-center justify-center rounded-[2.5rem]"
              style={{ backgroundColor: 'rgba(10, 11, 14, 0.6)', backdropFilter: 'blur(8px)' }}
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.9, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: 10 }}
                transition={{ duration: 0.2 }}
                className="w-64 p-4 rounded-2xl border border-white/10 shadow-2xl"
                style={{ backgroundColor: 'rgba(37, 40, 48, 0.97)', backdropFilter: 'blur(20px)' }}
              >
                {updateModal.status === 'available' && (
                  <>
                    <div className="flex items-center gap-2.5 mb-2">
                      <Download size={18} style={{ color: accentColor }} />
                      <span className="text-sm font-bold text-white">{t.updateAvailable(updateModal.version)}</span>
                    </div>
                    <p className="text-[10px] text-white/60 leading-snug mb-4">{t.updateDesc(updateModal.version)}</p>
                    <div className="flex gap-2">
                      <button onClick={() => setUpdateModal(null)}
                        className="flex-1 px-3 py-2 rounded-xl text-[10px] font-bold tracking-wide uppercase border border-white/15 text-white/80 hover:bg-white/10 transition-colors">
                        {t.later}
                      </button>
                      <button onClick={handleDownloadUpdate}
                        className="flex-1 px-3 py-2 rounded-xl text-[10px] font-bold tracking-wide uppercase transition-transform"
                        style={{ background: `linear-gradient(to right, ${accentColor}, ${accentColor}cc)`, color: timeOfDay === 'day' ? '#1a1a1a' : '#fff', boxShadow: `0 4px 15px ${accentColor}60` }}>
                        {t.downloadAndInstall}
                      </button>
                    </div>
                  </>
                )}
                {updateModal.status === 'downloading' && (
                  <div className="flex flex-col items-center py-1">
                    <Loader2 size={20} className="animate-spin mb-2" style={{ color: accentColor }} />
                    <span className="text-[11px] font-semibold text-white">{t.downloading(updateModal.progress)}</span>
                    <div className="w-full h-1.5 rounded-full bg-white/10 mt-3 overflow-hidden">
                      <motion.div className="h-full rounded-full" style={{ backgroundColor: accentColor }}
                        animate={{ width: `${updateModal.progress}%` }} transition={{ duration: 0.2 }} />
                    </div>
                  </div>
                )}
                {updateModal.status === 'installing' && (
                  <div className="flex flex-col items-center py-1">
                    <Loader2 size={20} className="animate-spin mb-2" style={{ color: accentColor }} />
                    <span className="text-[11px] font-semibold text-white">{t.installing}</span>
                  </div>
                )}
                {updateModal.status === 'error' && (
                  <>
                    <div className="flex items-center gap-2.5 mb-2">
                      <AlertCircle size={18} className="text-red-400" />
                      <span className="text-sm font-bold text-white">{t.updateError}</span>
                    </div>
                    <div className="flex gap-2">
                      <button onClick={() => setUpdateModal(null)}
                        className="flex-1 px-3 py-2 rounded-xl text-[10px] font-bold tracking-wide uppercase border border-white/15 text-white/80 hover:bg-white/10 transition-colors">
                        {t.later}
                      </button>
                      <button onClick={() => { openUrl('https://github.com/pavel807/TurnOffMachine/releases/latest'); setUpdateModal(null); }}
                        className="flex-1 px-3 py-2 rounded-xl text-[10px] font-bold tracking-wide uppercase transition-transform"
                        style={{ background: `linear-gradient(to right, ${accentColor}, ${accentColor}cc)`, color: timeOfDay === 'day' ? '#1a1a1a' : '#fff', boxShadow: `0 4px 15px ${accentColor}60` }}>
                        {t.openReleases}
                      </button>
                    </div>
                  </>
                )}
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Toast */}
        <AnimatePresence>
          {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
        </AnimatePresence>
        </>)}
      </div>
    </div>
  );
}