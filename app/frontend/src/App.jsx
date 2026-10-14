import React, { useEffect, useState } from 'react';
import { fetchModels, predict } from './api.js';
import ModelCard from './ModelCard.jsx';
import { SAMPLES } from './samples.js';

const DEFAULT_NOTICE =
  'Trained on synthetic data built for teaching. These labels say nothing about whether a real article is true.';

export default function App() {
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  const [models, setModels] = useState(null);
  const [maxChars, setMaxChars] = useState(6000);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);

  useEffect(() => {
    let alive = true;
    fetchModels()
      .then((m) => {
        if (!alive) return;
        setModels(m.models);
        setMaxChars(m.max_chars);
      })
      .catch((e) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, []);

  const length = text.trim().length;
  const tooLong = length > maxChars;
  const canSubmit = length > 0 && !tooLong && !busy;

  async function onSubmit(e) {
    e.preventDefault();
    if (!canSubmit) return;
    setBusy(true);
    setError(null);
    try {
      setResponse(await predict({ title, text }));
    } catch (err) {
      setResponse(null);
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function useSample(sample) {
    setTitle(sample.title);
    setText(sample.text);
    setResponse(null);
    setError(null);
  }

  const cards = response
    ? [...response.results.map((r) => ({ key: r.model, result: r })),
       ...response.unavailable.map((u) => ({
         key: u.model,
         name: (models && models.find((m) => m.id === u.model)?.name) || u.model,
         reason: u.reason,
       }))]
    : [];

  return (
    <div className="page">
      <div className="banner" role="note">
        Trained on synthetic data built for teaching; not a fact-checking tool.
      </div>

      <header className="header">
        <h1>Fake-news detection demo</h1>
        <p className="lead">
          Paste an article to see what three models trained on <strong>synthetic</strong> articles make of it:
          Logistic Regression and XGBoost on ant-colony-selected TF-IDF features, and a fine-tuned DistilBERT.
        </p>
      </header>

      <main>
        <form onSubmit={onSubmit} className="panel" aria-describedby="limits">
          <div className="field">
            <label htmlFor="title">Title (optional)</label>
            <input id="title" type="text" value={title} maxLength={300} onChange={(e) => setTitle(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="text">Article text</label>
            <textarea id="text" rows={8} value={text} onChange={(e) => setText(e.target.value)} />
            <p id="limits" className={`counter ${tooLong ? 'over' : ''}`}>
              {length} / {maxChars}
              {tooLong && ' (over the limit: shorten the text)'}
            </p>
          </div>
          <div className="actions">
            <button type="submit" className="primary" disabled={!canSubmit}>
              {busy ? 'Analysing…' : 'Analyse'}
            </button>
            {SAMPLES.map((s) => (
              <button key={s.id} type="button" className="secondary" onClick={() => useSample(s)}>
                {s.label}
              </button>
            ))}
          </div>
          <p className="muted small">
            The examples are invented text written in the style of the synthetic generator, not real news.
          </p>
        </form>

        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}

        {cards.length > 0 && (
          <section aria-label="Model results" className="results">
            <h2>What each model says</h2>
            <div className="cards">
              {cards.map((c) =>
                c.result ? (
                  <ModelCard key={c.key} result={c.result} />
                ) : (
                  <ModelCard key={c.key} name={c.name} unavailable={c.reason} />
                ),
              )}
            </div>
            <p className="muted small">{response.notice || DEFAULT_NOTICE}</p>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>
          This app runs locally. The GitHub Pages site for this project is a static results page and does not
          include it. Near-perfect or high scores on synthetic data are expected and do not carry over to real news.
        </p>
      </footer>
    </div>
  );
}
