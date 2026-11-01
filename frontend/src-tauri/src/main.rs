// Tauri desktop shell for Fiddlehead. The React frontend hosts all the UI and
// talks to the Python training engine over WebSocket (ws://localhost:8765).
//
// As a convenience, on launch we try to start that engine ourselves and stop it
// when the window closes — so `tauri dev` (and a packaged app that can locate
// the backend) is a genuine one-click experience. It's best-effort: if we can't
// find the backend or a Python interpreter, we do nothing and the UI simply
// shows its "start the engine" hint.
//
// Configure explicitly with env vars when needed:
//   FIDDLEHEAD_PYTHON  = path to a Python with the backend deps (torch, …)
//   FIDDLEHEAD_BACKEND = directory containing engine.py

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::env;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::{Manager, RunEvent};

struct Engine(Mutex<Option<Child>>);

fn find_backend() -> Option<PathBuf> {
    if let Ok(dir) = env::var("FIDDLEHEAD_BACKEND") {
        let p = PathBuf::from(dir);
        if p.join("engine.py").is_file() {
            return Some(p);
        }
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.to_path_buf());
        }
    }
    // dev layout (…/frontend/src-tauri) and common bundle layouts
    let rels = ["../../backend", "../backend", "backend", "../Resources/backend", "Resources/backend"];
    for root in roots {
        for rel in rels {
            let cand = root.join(rel);
            if cand.join("engine.py").is_file() {
                return cand.canonicalize().ok().or(Some(cand));
            }
        }
    }
    None
}

fn spawn_engine() -> Option<Child> {
    let backend = find_backend()?;
    let python = env::var("FIDDLEHEAD_PYTHON").unwrap_or_else(|_| "python3".to_string());
    match Command::new(&python).arg("engine.py").current_dir(&backend).spawn() {
        Ok(child) => {
            eprintln!("[fiddlehead] started engine: {} engine.py in {:?}", python, backend);
            Some(child)
        }
        Err(e) => {
            eprintln!("[fiddlehead] could not start engine ({}); connect to one instead: {}", python, e);
            None
        }
    }
}

fn main() {
    tauri::Builder::default()
        .manage(Engine(Mutex::new(spawn_engine())))
        .build(tauri::generate_context!())
        .expect("error while running Fiddlehead")
        .run(|app, event| {
            if let RunEvent::ExitRequested { .. } = event {
                if let Some(child) = app.state::<Engine>().0.lock().unwrap().as_mut() {
                    let _ = child.kill();
                }
            }
        });
}
