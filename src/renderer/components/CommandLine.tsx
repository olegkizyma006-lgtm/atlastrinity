/**
 * CommandLine - Bottom command input with integrated controls
 * Smart STT з аналізом типу мовлення
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';

// Типи мовлення з бекенду
type SpeechType = 'same_user' | 'new_phrase' | 'noise' | 'other_voice' | 'silence' | 'off_topic';

interface SmartSTTResponse {
  text: string;
  speech_type: SpeechType;
  confidence: number;
  combined_text: string;
  should_send: boolean;
  is_continuation: boolean;
}

interface CommandLineProps {
  onCommand: (command: string) => void;
  isVoiceEnabled?: boolean;
  onToggleVoice?: () => void;
  isProcessing?: boolean;
}

declare global {
  interface Window {
    volumeChecker: NodeJS.Timeout | number | null;
  }
}

const CommandLine: React.FC<CommandLineProps> = ({
  onCommand,
  isVoiceEnabled = true,
  onToggleVoice,
  isProcessing,
}) => {
  const [input, setInput] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [sttStatus, setSttStatus] = useState<string>(''); // Для показу статусу STT
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const silenceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pendingTextRef = useRef<string>(''); // Накопичений текст
  const isListeningRef = useRef<boolean>(false);
  const streamRef = useRef<MediaStream | null>(null);
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const maxVolumeRef = useRef<number>(0); // Для простого VAD
  const audioContextRef = useRef<AudioContext | null>(null);

  // Auto-expand logic
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening();
    };
  }, []);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (input.trim()) {
      onCommand(input.trim());
      setInput('');
      pendingTextRef.current = '';
      if (textareaRef.current) textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Планування автоматичної відправки після мовчання
  // ВАЖЛИВО: ця функція повинна бути визначена ПЕРЕД handleSTTResponse
  const scheduleSend = useCallback(() => {
    // Скидаємо попередній таймер
    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current);
    }

    silenceTimeoutRef.current = setTimeout(() => {
      const textToSend = pendingTextRef.current.trim();

      if (textToSend) {
        setSttStatus('🚀 Відправка...');
        onCommand(textToSend);
        setInput('');
        pendingTextRef.current = '';

        // Миттєве відновлення статусу "Слухаю", оскільки Full Duplex працює
        if (isListeningRef.current) {
          setSttStatus('🎙️ Слухаю...');
        } else {
          setSttStatus('');
        }

        if (textareaRef.current) textareaRef.current.style.height = 'auto';
      }
    }, 6000); // 6 секунд мовчання перед відправкою, як домовлено з користувачем
  }, [onCommand]);

  // Обробка відповіді від Smart STT
  const handleSTTResponse = useCallback(
    (data: SmartSTTResponse) => {
      const { speech_type, combined_text, text, should_send } = data;

      // Якщо сервер явно каже відправити (наприклад, довга пауза на сервері)
      if (should_send && combined_text.trim()) {
        pendingTextRef.current = combined_text;
        scheduleSend(); // Виклик негайно (через існуючий механізм таймера з 0 затримкою або просто викликати логіку submit)
        // Насправді краще просто форсувати таймер
        if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);
        const textToSend = combined_text.trim();
        setSttStatus('🚀 Відправка (Server Trigger)...');
        onCommand(textToSend);
        setInput('');
        pendingTextRef.current = '';
        setSttStatus('🎙️ Слухаю...');
        return;
      }

      switch (speech_type) {
        case 'silence':
          setSttStatus('Mw... (Тиша)');
          // При тиші відправляємо накопичений текст
          if (pendingTextRef.current.trim()) {
            scheduleSend();
          }
          break;

        case 'noise':
          // setSttStatus('🔊 ...'); // Менш нав'язливий статус для шуму
          break;

        case 'other_voice':
          setSttStatus('👤 Інший голос');
          break;

        case 'off_topic':
          setSttStatus('💬 ...');
          break;

        case 'same_user':
        case 'new_phrase':
          // Оновлюємо текст
          if (text && text.trim()) {
            pendingTextRef.current = combined_text;
            setInput(combined_text);
            setSttStatus(`✅ ${text.slice(0, 30)}...`);

            // Перезапускаємо таймер для відправки
            scheduleSend();
          } else {
            setSttStatus('✅ ...');
          }
          break;

        default:
          // console.warn('Unknown speech type:', speech_type);
          break;
      }
    },
    [scheduleSend]
  );

  // Відправка аудіо на розумний STT
  const processAudioChunk = useCallback(
    async (audioBlob: Blob) => {
      // console.log('🎤 Processing audio chunk:', audioBlob.size, 'bytes, type:', audioBlob.type);

      // Визначаємо розширення файлу
      let fileExtension = 'wav';
      if (audioBlob.type.includes('webm')) {
        fileExtension = 'webm';
      } else if (audioBlob.type.includes('ogg')) {
        fileExtension = 'ogg';
      }

      const formData = new FormData();
      formData.append('audio', audioBlob, `recording.${fileExtension}`);
      formData.append('previous_text', pendingTextRef.current);

      try {
        // console.log('📤 Sending to STT server...');
        const response = await fetch('http://127.0.0.1:8000/api/stt/smart', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const data: SmartSTTResponse = await response.json();
          // console.log('🎤 Smart STT Response:', data);

          handleSTTResponse(data);
        } else {
          const errorText = await response.text();
          console.error('❌ STT server error:', response.status, response.statusText, errorText);
          setSttStatus('❌ Помилка STT');
        }
      } catch (error) {
        console.error('❌ Smart STT error:', error);
        setSttStatus("❌ Помилка з'єднання");
      }
    },
    [handleSTTResponse]
  );

  // Початок запису
  const startListening = async () => {
    try {
      // console.log('🎙️ Starting to listen...');

      // Якщо TTS вимкнено, автоматично вмикаємо
      if (!isVoiceEnabled && onToggleVoice) {
        // console.log('🔊 Enabling voice...');
        onToggleVoice();
      }

      // Отримуємо stream
      let stream = streamRef.current;
      if (!stream || !stream.active) {
        // console.log('🎤 Requesting microphone access...');
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: false,
            autoGainControl: true, // Вмикаємо автоматичну регулювання гучності
            sampleRate: 48000,
          },
        });
        streamRef.current = stream;
        // console.log('✅ Microphone access granted, stream active:', stream.active);


        // Перевіряємо гучність
        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);

        const checkVolume = () => {
          const dataArray = new Uint8Array(analyser.frequencyBinCount);
          analyser.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
          if (average > maxVolumeRef.current) {
            maxVolumeRef.current = average;
          }
          // console.log('🔊 Audio level:', average);
        };

        // Перевіряємо гучність кожні 100мс
        const volumeChecker = setInterval(checkVolume, 100);

        // Зберігаємо для очищення
        window.volumeChecker = volumeChecker;
      } else {
        // console.log('♻️ Reusing existing stream');
      }

      // КРИТИЧНО: оновлюємо ref СИНХРОННО перед викликом startRecordingCycle
      isListeningRef.current = true;
      setIsListening(true);
      setSttStatus('🎙️ Слухаю...');
      // console.log('🎙️ Started listening, isListeningRef:', isListeningRef.current);

      // Запускаємо циклічний запис (2 секунди на чанк для Smart STT)
      startRecordingCycle();
    } catch (error) {
      console.error('❌ Microphone access error:', error);
      // При помилці скидаємо стан
      isListeningRef.current = false;
      setIsListening(false);
      setSttStatus('');
      handleMicError(error);
    }
  };

  // Циклічний запис
  const startRecordingCycle = () => {
    // console.log(
    //   '🔄 Starting recording cycle, isListening:',
    //   isListeningRef.current,
    //   'stream active:',
    //   streamRef.current?.active
    // );
    if (!streamRef.current?.active || !isListeningRef.current) return;

    // Примусово використовуємо WAV формат
    let mimeType = 'audio/webm';
    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
      mimeType = 'audio/webm;codecs=opus';
    } else if (MediaRecorder.isTypeSupported('audio/wav')) {
      mimeType = 'audio/wav';
    }

    // console.log('🎤 Using MIME type:', mimeType);
    const mediaRecorder = new MediaRecorder(streamRef.current, { mimeType });
    mediaRecorderRef.current = mediaRecorder;
    audioChunksRef.current = [];
    maxVolumeRef.current = 0; // Скидаємо перед новим чанком

    mediaRecorder.ondataavailable = (event) => {
      // console.log('📊 Audio data available:', event.data.size, 'bytes');
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      // console.log(
      //   '🛑 MediaRecorder stopped, chunks:',
      //   audioChunksRef.current.length,
      //   'max volume:',
      //   maxVolumeRef.current
      // );

      // Простий VAD: якщо було дуже тихо (тиша/шум), не відправляємо
      if (maxVolumeRef.current < 12) {
        // console.log('🔇 Chunk too quiet, skipping STT');
        setSttStatus('🔇 Тиша...');
      } else if (audioChunksRef.current.length > 0) {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        // console.log('🎤 Created audio blob:', mimeType, audioBlob.size, 'bytes');
        await processAudioChunk(audioBlob);
      } else {
        console.log('⚠️ No audio chunks recorded');
      }

      // Продовжуємо цикл якщо ще слухаємо
      if (isListeningRef.current && streamRef.current?.active) {
        // console.log('🔄 Continuing recording cycle...');
        startRecordingCycle();
      }
    };

    // console.log('▶️ Starting MediaRecorder...');
    mediaRecorder.start();

    // Зупиняємо запис через 2 секунди для обробки (оптимально для частоти оновлень)
    recordingIntervalRef.current = setTimeout(() => {
      // console.log('⏱️ Stopping MediaRecorder after 2 seconds...');
      if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
    }, 2000);
  };

  // Зупинка прослуховування
  const stopListening = () => {
    // console.log('🛑 Stopping listening');

    // КРИТИЧНО: оновлюємо ref СИНХРОННО
    isListeningRef.current = false;
    setIsListening(false);
    setSttStatus('');

    // Зупиняємо перевірку гучності
    if (window.volumeChecker) {
      clearInterval(window.volumeChecker as number);
      window.volumeChecker = null;
    }

    if (recordingIntervalRef.current) {
      clearTimeout(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current);
      silenceTimeoutRef.current = null;
    }

    // Зупиняємо stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    // Закриваємо AudioContext
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(console.error);
      audioContextRef.current = null;
    }
  };

  // Обробка помилок мікрофона
  const handleMicError = (error: unknown) => {
    if (error instanceof DOMException) {
      switch (error.name) {
        case 'NotFoundError':
        case 'DevicesNotFoundError':
          alert(
            '❌ Мікрофон не знайдено\n\nПеревірте:\n• Мікрофон підключений\n• Мікрофон увімкнений в системі'
          );
          break;
        case 'NotAllowedError':
        case 'PermissionDeniedError':
          alert('❌ Доступ заблоковано\n\nДозвольте доступ до мікрофона');
          break;
        case 'NotReadableError':
        case 'TrackStartError':
          alert('❌ Мікрофон зайнятий\n\nЗакрийте інші програми');
          break;
        default:
          alert(`❌ Помилка: ${error.message}`);
      }
    }
  };

  const toggleListening = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  return (
    <div className="command-line-container font-mono">
      <div className="flex items-baseline gap-2 pt-2 bg-transparent pb-0">
        {/* Left Controls - Only TTS */}
        <div className="flex items-center gap-1">
          {/* TTS Toggle */}
          <button
            onClick={onToggleVoice}
            className={`control-btn ${isVoiceEnabled ? 'active' : ''}`}
            title="Toggle Voice (TTS)"
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
              {isVoiceEnabled ? (
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
              ) : (
                <line x1="23" y1="9" x2="17" y2="15"></line>
              )}
            </svg>
          </button>
        </div>

        {/* Input Field with STT Status */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="ENTER_CORE_COMMAND..."
            className="command-textarea-extended"
            spellCheck={false}
            rows={1}
            autoFocus
          />
          <div className="absolute right-3 bottom-2 flex items-center gap-2">
            {(isProcessing || sttStatus) && (
              <span className={`text-[9px] tracking-wider animate-pulse ${isProcessing ? 'text-amber-400' : 'text-cyan-400/70'}`}>
                {isProcessing ? '🤔 Думаю...' : sttStatus}
              </span>
            )}
            <span className="text-blue-500/20 text-[9px] pointer-events-none tracking-widest">
              {input.length > 0 ? 'ENTER ' : ''}⏎
            </span>
          </div>
        </div>

        {/* Right Controls - Send and Mic */}
        <div className="flex items-center gap-2">
          {/* STT/Mic Button */}
          <button
            onClick={toggleListening}
            className={`control-btn ${isListening ? 'listening' : ''}`}
            title="Toggle Smart Mic (STT)"
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              {isListening ? (
                <>
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                  <line x1="12" y1="19" x2="12" y2="23"></line>
                  <line x1="8" y1="23" x2="16" y2="23"></line>
                </>
              ) : (
                <>
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                  <line x1="12" y1="19" x2="12" y2="23"></line>
                  <line x1="8" y1="23" x2="16" y2="23"></line>
                  <line x1="1" y1="1" x2="23" y2="23" strokeOpacity="1"></line>
                </>
              )}
            </svg>
          </button>

          {/* Send Button */}
          <button
            onClick={() => handleSubmit()}
            disabled={!input.trim()}
            className={`send-btn ${input.trim() ? 'active' : ''}`}
            title="Send Command (Enter)"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ transform: 'rotate(45deg) translateY(-2px) translateX(2px)' }}
            >
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default CommandLine;
