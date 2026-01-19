/**
 * AtlasTrinity - Electron Main Process
 */

import { app, BrowserWindow, ipcMain, systemPreferences } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { spawn, ChildProcess } from 'child_process';
import { checkPermissions, requestPermissions } from './permissions';

let mainWindow: BrowserWindow | null = null;

const isDev = process.env.NODE_ENV === 'development';

// Check if first-run setup is needed
function isFirstRun(): boolean {
  const setupMarker = path.join(os.homedir(), '.config/atlastrinity/setup_complete');
  return !fs.existsSync(setupMarker);
}

// Run first-run setup in a terminal window
async function runFirstRunSetup(): Promise<boolean> {
  // Find Python executable
  let pythonExec = 'python3';
  const bundledPython = path.join(process.resourcesPath, '.venv/bin/python');
  if (fs.existsSync(bundledPython)) {
    pythonExec = bundledPython;
  }

  console.log('[SETUP] Running first-run installer...');

  return new Promise((resolve) => {
    const setupProcess = spawn(pythonExec, ['-m', 'brain.first_run_installer'], {
      cwd: process.resourcesPath,
      stdio: 'inherit', // Show output in terminal
      env: {
        ...process.env,
        PYTHONPATH: process.resourcesPath,
        PATH: `${path.dirname(pythonExec)}:/opt/homebrew/bin:${process.env.PATH}`,
      },
    });

    setupProcess.on('close', (code: number) => {
      console.log(`[SETUP] First-run installer exited with code ${code}`);
      resolve(code === 0);
    });

    setupProcess.on('error', (err: Error) => {
      console.error('[SETUP] Failed to run installer:', err);
      resolve(false);
    });
  });
}

async function createWindow(): Promise<void> {
  // First-run setup (production only)
  if (!isDev && isFirstRun()) {
    console.log('[SETUP] First run detected, launching setup...');
    const setupSuccess = await runFirstRunSetup();
    if (!setupSuccess) {
      console.error('[SETUP] First-run setup failed, but continuing...');
    }
  }

  // Check required permissions first
  const permissionsOk = await checkPermissions();
  if (!permissionsOk) {
    await requestPermissions();
  }

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    transparent: true,
    vibrancy: 'under-window',
    visualEffectState: 'active',
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#00000000',
    webPreferences: {
      devTools: isDev,
    },
  });

  // Load the app
  if (isDev) {
    await mainWindow.loadURL('http://localhost:3000');
    mainWindow.webContents.openDevTools();

    // В dev режимі Python сервер запускається через npm run dev:brain
    // НЕ запускаємо тут, щоб уникнути конфлікту на порту 8000
    // startPythonServerDev();
  } else {
    await mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));

    // Spawn Python Server in Production
    startPythonServer();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Protect against focus loss causing black screen
  // Вимкнено, щоб не перемикати фокус автоматично
  // mainWindow.on('blur', () => {
  //     console.log('[ELECTRON] Window lost focus - monitoring for recovery');
  //
  //     // Set timeout to restore focus if needed
  //     setTimeout(() => {
  //         if (mainWindow && !mainWindow.isFocused()) {
  //             console.log('[ELECTRON] Auto-restoring window focus');
  //             mainWindow.focus();
  //         }
  //     }, 500);
  // });

  // Protect renderer from hanging
  mainWindow.webContents.on('unresponsive', () => {
    console.error('[ELECTRON] Renderer became unresponsive - attempting reload');
    if (mainWindow) {
      mainWindow.webContents.reload();
    }
  });

  mainWindow.webContents.on('responsive', () => {
    console.log('[ELECTRON] Renderer became responsive again');
  });
}

// Python Server Management
let pythonProcess: ChildProcess | null = null;

/*
function startPythonServerDev() {
    // В dev режимі використовуємо .venv з проекту
    const projectRoot = path.join(__dirname, '../../..');
    const venvPython = path.join(projectRoot, '.venv/bin/python');

    let pythonExec = 'python3';
    if (fs.existsSync(venvPython)) {
        pythonExec = venvPython;
    }

    console.log(`[DEV] Starting Python server with: ${pythonExec}`);
    console.log(`[DEV] Working directory: ${projectRoot}`);

    pythonProcess = spawn(pythonExec, ['-m', 'uvicorn', 'brain.server:app', '--host', '127.0.0.1', '--port', '8000', '--reload'], {
        cwd: projectRoot,
        env: {
            ...process.env,
            PYTHONUNBUFFERED: '1',
            PYTHONPATH: path.join(projectRoot, 'src'),
        }
    });

    pythonProcess.stdout?.on('data', (data: Buffer) => {
        const message = data.toString();
        console.log(`[Python]: ${message}`);
    });

    pythonProcess.stderr?.on('data', (data: Buffer) => {
        const message = data.toString();
        // uvicorn пише в stderr
        console.log(`[Python]: ${message}`);
    });

    pythonProcess.on('error', (err: Error) => {
        console.error(`[DEV] Failed to start Python server: ${err}`);
    });

    pythonProcess.on('close', (code: number | null) => {
        console.log(`[DEV] Python process exited with code ${code}`);
    });
}
*/

function startPythonServer() {
  // Robust Python discovery
  let pythonExec = process.env.ATLASTRINITY_PYTHON || 'python3'; // Default fallback

  // 1. Try app-local .venv (dev/local prod test)
  const projectVenvPath = path.join(app.getAppPath(), '.venv/bin/python');

  // 2. Try packaged .venv if it was bundled (Portable mode)
  // When moved to /Applications, app.getAppPath() points to the app content
  const bundledVenvPath = path.join(process.resourcesPath, '.venv/bin/python');

  if (fs.existsSync(bundledVenvPath)) {
    pythonExec = bundledVenvPath;
  } else if (fs.existsSync(projectVenvPath)) {
    pythonExec = projectVenvPath;
  }

  console.log(`Starting Python server with: ${pythonExec}`);
  console.log(`Working directory: ${process.resourcesPath}`);

  pythonProcess = spawn(pythonExec, ['-m', 'brain.server'], {
    cwd: process.resourcesPath,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      PYTHONPATH: process.resourcesPath,
      PATH: `${path.dirname(pythonExec)}:${process.env.PATH}`, // Add venv/bin to PATH
    },
  });

  pythonProcess.stdout?.on('data', (data: Buffer) => {
    const message = data.toString();
    console.log(`[Python]: ${message}`);
    if (mainWindow) {
      mainWindow.webContents.executeJavaScript(
        `console.log('[Python]: ' + ${JSON.stringify(message)})`
      );
    }
  });

  pythonProcess.stderr?.on('data', (data: Buffer) => {
    const message = data.toString();
    console.error(`[Python Err]: ${message}`);
    if (mainWindow) {
      mainWindow.webContents.executeJavaScript(
        `console.error('[Python Err]: ' + ${JSON.stringify(message)})`
      );
    }
  });

  pythonProcess.on('error', (err: Error) => {
    console.error(`Failed to start Python server: ${err}`);
    if (mainWindow) {
      mainWindow.webContents.executeJavaScript(
        `console.error('CRITICAL: Failed to start Python server: ' + ${JSON.stringify(err.message)})`
      );
    }
  });

  pythonProcess.on('close', (code: number | null) => {
    console.log(`Python process exited with code ${code}`);
    if (mainWindow && code !== 0 && code !== null) {
      mainWindow.webContents.executeJavaScript(
        `console.error('CRITICAL: Python server exited with code ' + ${code})`
      );
    }
  });
}

// IPC Handlers for renderer communication
ipcMain.handle('get-system-info', async () => {
  return {
    platform: process.platform,
    arch: process.arch,
    version: app.getVersion(),
  };
});

ipcMain.handle('check-accessibility', async () => {
  return systemPreferences.isTrustedAccessibilityClient(false);
});

ipcMain.handle('request-accessibility', async () => {
  return systemPreferences.isTrustedAccessibilityClient(true);
});

// App lifecycle
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  // В development режимі завжди закриваємо додаток (включно з macOS)
  // В production на macOS - стандартна поведінка (додаток залишається в dock)
  if (isDev || process.platform !== 'darwin') {
    app.quit();
  }
});

// Ensure Python server is killed when app quits
app.on('before-quit', () => {
  if (pythonProcess) {
    console.log('Quitting: Killing Python server...');
    // Send SIGTERM to allow graceful shutdown via lifespan manager
    pythonProcess.kill('SIGTERM');
    pythonProcess = null;
  }

  // Clean up any stray processes (dev always aggressive, production as safety net)
  console.log('Final process cleanup...');
  const { execSync } = require('child_process');
  try {
    // Targeted pkill for core components to avoid "orphans"
    // brain.server handles its own children, but this is a final fail-safe
    const targets = "vibe_server vibe brain.server mcp-server memory_server graph_server macos-use";
    const command = targets.split(' ').map(t => `pkill -9 -f "${t}"`).join('; ') + '; true';

    execSync(command, {
      stdio: 'ignore',
      timeout: 5000
    });
    console.log('Cleanup completed.');
  } catch (e) {
    console.log('Cleanup finished.');
  }
});

app.on('will-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill('SIGKILL');
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// Handle certificate errors in development
if (isDev) {
  app.on('certificate-error', (event, webContents, url, error, certificate, callback) => {
    event.preventDefault();
    callback(true);
  });
}
