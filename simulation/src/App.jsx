import { useEffect, useMemo, useState } from "react";
import { sourceManifest } from "./data/sourceManifest";
import { runTrace } from "./data/runTrace";
import { trainTrace } from "./data/trainTrace";

const traces = { run: runTrace, train: trainTrace };

const icon = {
  play: "▶",
  pause: "Ⅱ",
  next: "›",
  previous: "‹",
  reset: "↺",
};

function formatValue(value) {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function CodeViewer({ current }) {
  const source = sourceManifest[current.file] || { lines: [] };
  const lineNumber = current.line || 1;
  return (
    <section className="panel code-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">SOURCE</span>
          <h2>{current.file}</h2>
        </div>
        <span className="function-pill">{current.functionName}</span>
      </div>
      <div className="code-scroll">
        {source.lines.map((line, index) => {
          const number = index + 1;
          const active = number === lineNumber;
          return (
            <div className={`code-line ${active ? "active" : ""}`} key={`${current.file}-${number}`}>
              <span className="line-number">{number}</span>
              <span className="line-code">{line || " "}</span>
              {active && <span className="line-marker">●</span>}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function Timeline({ trace, currentIndex, onSelect }) {
  return (
    <aside className="timeline panel">
      <div className="panel-heading compact">
        <div>
          <span className="eyebrow">EXECUTION TRACE</span>
          <h2>Call timeline</h2>
        </div>
        <span className="step-count">{currentIndex + 1}/{trace.length}</span>
      </div>
      <div className="timeline-list">
        {trace.map((item, index) => (
          <button
            className={`timeline-item ${index === currentIndex ? "selected" : ""} ${index < currentIndex ? "visited" : ""}`}
            key={item.id}
            onClick={() => onSelect(index)}
          >
            <span className={`trace-dot ${item.category}`} />
            <span className="timeline-copy">
              <strong>{item.title}</strong>
              <small>{item.file}:{item.line}</small>
            </span>
            <span className="timeline-index">{String(index + 1).padStart(2, "0")}</span>
          </button>
        ))}
      </div>
    </aside>
  );
}

function Inspector({ current }) {
  const sections = [
    ["Arguments", current.arguments],
    ["Locals", current.locals],
    ["Return", current.returnValue],
  ];
  return (
    <section className="panel inspector">
      <div className="panel-heading compact">
        <div>
          <span className="eyebrow">INSPECTOR</span>
          <h2>Frame state</h2>
        </div>
        <span className={`category-badge ${current.category}`}>{current.category}</span>
      </div>
      <div className="explanation">
        <span className="explanation-icon">i</span>
        <p>{current.explanation}</p>
      </div>
      {sections.map(([label, value]) => (
        <div className="inspector-section" key={label}>
          <div className="inspector-label">{label}</div>
          <pre>{formatValue(value)}</pre>
        </div>
      ))}
      <div className="inspector-section">
        <div className="inspector-label">Call stack</div>
        <div className="stack-list">
          {current.callStack.map((frame, index) => (
            <div className={`stack-frame ${index === current.callStack.length - 1 ? "top" : ""}`} key={frame + index}>
              <span>{index === current.callStack.length - 1 ? "◆" : "◇"}</span>{frame}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Console({ output }) {
  return (
    <section className="console panel">
      <div className="console-title"><span className="eyebrow">TERMINAL</span><span>python simulator</span></div>
      <div className="console-lines">
        {output.length ? output.map((line, index) => <div key={index}><span className="prompt">&gt;</span>{line}</div>) : <div className="muted"><span className="prompt">&gt;</span>En attente d'une sortie...</div>}
      </div>
    </section>
  );
}

function PacmanBoard({ state }) {
  const cell = (value, x, y) => {
    const pac = Math.round(state.pacman.x) === x && Math.round(state.pacman.y) === y;
    const ghost = Math.round(state.ghost.x) === x && Math.round(state.ghost.y) === y;
    if (pac) return <span className={`entity pacman direction-${state.pacman.direction.toLowerCase()}`}>◕</span>;
    if (ghost) return <span className={`entity ghost ${state.ghost.scared ? "scared" : ""}`}>♟</span>;
    if (value === "%") return <span className="wall"> </span>;
    if (value === ".") return <span className="food-dot">·</span>;
    if (value === "o") return <span className="capsule">◦</span>;
    return <span> </span>;
  };
  return (
    <section className="panel board-panel">
      <div className="panel-heading compact">
        <div><span className="eyebrow">RUNTIME STATE</span><h2>Pacman board</h2></div>
        <span className="turn-badge">TURN {state.turn}</span>
      </div>
      <div className="board-wrap">
        <div className="board">
          {state.board.map((row, y) => [...row].map((value, x) => <div className="cell" key={`${x}-${y}`}>{cell(value, x, y)}</div>))}
        </div>
      </div>
      <div className="board-stats">
        <div><span>Score</span><strong>{state.score}</strong></div>
        <div><span>Food left</span><strong>{state.food}</strong></div>
        <div><span>Pacman</span><strong>({state.pacman.x}, {state.pacman.y})</strong></div>
        <div><span>Ghost</span><strong>{state.ghost.scared ? "scared" : "chasing"}</strong></div>
      </div>
    </section>
  );
}

function ActionScores({ current }) {
  const logits = current.locals?.logits;
  const candidates = current.locals?.candidates;
  if (!logits && !candidates) return null;
  const values = logits || candidates;
  return (
    <section className="panel scores-panel">
      <div className="panel-heading compact"><div><span className="eyebrow">MODEL OUTPUT</span><h2>Action logits</h2></div><span className="tensor-tag">Tensor[1,5]</span></div>
      <div className="score-list">
        {Object.entries(values).map(([action, score]) => {
          const numeric = Number(score);
          const max = Math.max(...Object.values(values).map(Number));
          return <div className={`score-row ${action === current.locals?.best_action ? "chosen" : ""}`} key={action}>
            <span className="score-action">{action}</span>
            <div className="score-track"><i style={{ width: `${Math.max(8, ((numeric + 1) / 2.5) * 100)}%` }} /></div>
            <b>{numeric.toFixed(2)}</b>
            {action === current.locals?.best_action && <span className="chosen-label">SELECTED</span>}
          </div>;
        })}
      </div>
    </section>
  );
}

function TrainingPanel({ state }) {
  const metrics = state.epochMetrics || [];
  const maxLoss = Math.max(...metrics.map((m) => m.trainLoss), 1);
  return (
    <section className="panel training-panel">
      <div className="panel-heading compact">
        <div><span className="eyebrow">TRAINING TELEMETRY</span><h2>Learning curve</h2></div>
        <span className="turn-badge">EPOCH {state.activeEpoch || 1}</span>
      </div>
      <div className="chart">
        {metrics.map((metric) => (
          <div className="chart-column" key={metric.epoch} title={`Epoch ${metric.epoch}: val acc ${metric.valAcc}%`}>
            <div className="bar train-bar" style={{ height: `${Math.max(10, metric.trainLoss / maxLoss * 100)}%` }} />
            <div className="bar val-bar" style={{ height: `${Math.max(10, metric.valLoss / maxLoss * 100)}%` }} />
            <span>{metric.epoch}</span>
          </div>
        ))}
      </div>
      <div className="legend"><span><i className="legend-dot train" />train loss</span><span><i className="legend-dot val" />validation loss</span></div>
      <div className="metric-grid">
        {metrics.slice(-1).map((m) => <div className="metric-card" key={m.epoch}><span>VAL ACC</span><strong>{m.valAcc}%</strong><small>lr {m.lr}</small></div>)}
        <div className="metric-card"><span>BEST MODEL</span><strong>86.4%</strong><small>pacman_model.pth</small></div>
      </div>
    </section>
  );
}

function App() {
  const [mode, setMode] = useState("run");
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const trace = traces[mode];
  const current = trace[index];
  const output = useMemo(() => current.consoleOutput || [], [current]);

  useEffect(() => {
    setIndex(0);
    setPlaying(false);
  }, [mode]);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => {
      setIndex((value) => {
        if (value >= trace.length - 1) {
          setPlaying(false);
          return value;
        }
        return value + 1;
      });
    }, 1300 / speed);
    return () => window.clearInterval(timer);
  }, [playing, speed, trace.length]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      if (event.key === "ArrowRight") move(1);
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === " ") {
        event.preventDefault();
        setPlaying((value) => !value);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  const move = (delta) => {
    setPlaying(false);
    setIndex((value) => Math.max(0, Math.min(trace.length - 1, value + delta)));
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark">◕</div><div><strong>Pacman Python</strong><span>DEBUGGER / SIMULATION</span></div></div>
        <div className="mode-switch">
          <button className={mode === "run" ? "active" : ""} onClick={() => setMode("run")}><span>01</span>run.py</button>
          <button className={mode === "train" ? "active" : ""} onClick={() => setMode("train")}><span>02</span>train.py</button>
        </div>
        <div className="top-status"><span className="live-dot" />STATIC TRACE <span className="divider" /> PYTHON 3.8</div>
      </header>

      <section className="controlbar">
        <div className="breadcrumb"><span>PROJECT</span><b>/</b><strong>{mode === "run" ? "run.py" : "train.py"}</strong><b>/</b><span>{current.functionName}</span></div>
        <div className="transport">
          <button onClick={() => move(-1)} aria-label="Previous step">{icon.previous}</button>
          <button className="play-button" onClick={() => setPlaying((value) => !value)} aria-label="Play or pause">{playing ? icon.pause : icon.play}</button>
          <button onClick={() => move(1)} aria-label="Next step">{icon.next}</button>
          <button onClick={() => { setPlaying(false); setIndex(0); }} aria-label="Reset">{icon.reset}</button>
          <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value="0.5">0.5x</option><option value="1">1x</option><option value="2">2x</option><option value="4">4x</option></select>
        </div>
        <div className="progress-copy"><strong>{String(index + 1).padStart(2, "0")}</strong><span>/ {String(trace.length).padStart(2, "0")}</span><div className="progress-track"><i style={{ width: `${((index + 1) / trace.length) * 100}%` }} /></div></div>
      </section>

      <div className="workspace">
        <Timeline trace={trace} currentIndex={index} onSelect={(value) => { setPlaying(false); setIndex(value); }} />
        <div className="center-column">
          <div className="step-banner"><div className={`step-icon ${current.category}`}>{current.category === "network" ? "⌁" : current.category === "decision" ? "✦" : "›"}</div><div><span className="eyebrow">CURRENT OPERATION</span><h1>{current.title}</h1></div><div className="step-file">{current.file}:{current.line}</div></div>
          <CodeViewer current={current} />
          <Console output={output} />
        </div>
        <div className="right-column">
          <Inspector current={current} />
          {mode === "run" ? <><PacmanBoard state={current.visualState} /><ActionScores current={current} /></> : <TrainingPanel state={current.visualState} />}
        </div>
      </div>
      <footer className="footer"><span><kbd>←</kbd><kbd>→</kbd> navigate</span><span><kbd>Space</kbd> play / pause</span><span>Trace pédagogique — lignes source et états simulés</span></footer>
    </main>
  );
}

export default App;
