// Source of assets/app.js. Rebuild with: node scripts/build_jsx.mjs (see README).
// Reads the files written by the notebook and scripts (outputs/*.json, outputs/selected_features.txt).
const { useState, useEffect, useRef } = React;

const THEMES = [
  ['modern', 'Modern'], ['git', 'Git Dark'], ['rainbow', 'Rainbow Gradient'], ['neon', 'Neon'],
  ['pastel', 'Pastel'], ['monochrome', 'Monochrome'], ['cyberpunk', 'Cyberpunk'], ['retro', 'Retro'],
];

function Chart({ figure, theme }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!figure || !ref.current) return;
    const color = getComputedStyle(ref.current).color;
    const layout = Object.assign({}, figure.layout, {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color },
      autosize: true, margin: { l: 50, r: 10, t: 70, b: 90 },
    });
    // Plotly does not wrap titles, so break long ones before the parenthesis and left-align them
    if (layout.title && layout.title.text) {
      layout.title = Object.assign({}, layout.title, {
        text: layout.title.text.replace(' (', '<br>('), x: 0.02, xanchor: 'left', font: { size: 15 },
      });
    }
    delete layout.template;
    Plotly.react(ref.current, figure.data, layout, { responsive: true, displayModeBar: false });
  }, [figure, theme]);
  return <div ref={ref} className="chart" role="img" aria-label={figure && figure.layout.title ? figure.layout.title.text : 'chart'}></div>;
}

const fmt = (x) => x.toFixed(4);
const fmtTime = (s) => (s < 1 ? (s * 1000).toFixed(0) + ' ms' : s.toFixed(1) + ' s');

function MetricsTable({ metrics }) {
  return (
    <div>
      <p className="text-primary note">{metrics.label}</p>
      <div className="table-wrap">
        <table className="metrics text-primary">
          <caption>Test split of {metrics.test_articles.toLocaleString()} synthetic articles; precision, recall and F1 are macro-averaged. Majority-class baseline accuracy: {fmt(metrics.baseline_accuracy)}.</caption>
          <thead>
            <tr><th scope="col">Model</th><th scope="col">Accuracy</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1</th><th scope="col">Training time</th><th scope="col">Inference time (test split)</th></tr>
          </thead>
          <tbody>
            {metrics.models.map(m => (
              <tr key={m.id}>
                <th scope="row">{m.name}</th>
                <td>{fmt(m.accuracy)}</td><td>{fmt(m.precision_macro)}</td><td>{fmt(m.recall_macro)}</td><td>{fmt(m.f1_macro)}</td>
                <td>{fmtTime(m.train_seconds)}</td><td>{fmtTime(m.inference_seconds)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="status-list text-primary">
        {metrics.models.map(m => <li key={m.id}><strong>{m.name}:</strong> {m.status}. {m.note}</li>)}
      </ul>
    </div>
  );
}

function App() {
  const [theme, setTheme] = useState('modern');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [conceptStep, setConceptStep] = useState(0);

  useEffect(() => {
    const get = (url, kind) => fetch(url).then(res => {
      if (!res.ok) throw new Error(url + ': HTTP ' + res.status);
      return kind === 'text' ? res.text() : res.json();
    });
    Promise.all([
      get('outputs/mi_scores.json'), get('outputs/model_comparison.json'), get('outputs/selected_features.txt', 'text'),
      get('outputs/model_metrics.json').catch(() => null), // optional: older output folders do not have it
    ]).then(([mi, comparison, features, metrics]) => setResults({ mi, comparison, metrics, features: features.split(/\r?\n/).filter(Boolean) }))
      .catch(err => setError(String(err.message || err)));
  }, []);

  const conceptSteps = [
    { title: 'What is fake news detection?',
      content: 'Fake news detection tries to label articles as real or fake from their text and metadata using ML and NLP. In this demo the articles are synthetic, so the labels come from a writing-style recipe, not from checking facts.' },
    { title: 'Ant Colony Optimization (ACO)',
      content: 'ACO mimics ant foraging to pick useful features. Each "ant" samples a set of features, favouring those with high mutual information and strong pheromone, and features in well-scoring sets get more pheromone. Scores come from a validation split of the training data only.' },
    { title: 'Clustering-based mutual information',
      content: 'Each feature\'s values are grouped into 3 clusters, and mutual information measures how much the cluster tells you about the label. It is a quick filter that can pick up non-linear patterns.' },
    { title: 'Embedded-filter ACO',
      content: 'It combines the speed of the filter (clustering-based MI) with the search of a wrapper (ACO). The search is limited to 50 to 200 features per ant.' },
  ];

  return (
    <div className={theme + '-theme page'}>
      <div className="banner" role="note">Trained on synthetic data built for teaching; not a fact-checking tool.</div>
      <header className="bar">
        <h1 className="text-primary">Fake News Detection Explorer</h1>
        <label className="theme-picker text-primary">
          <span>Theme</span>
          <select value={theme} onChange={e => setTheme(e.target.value)} className="bg-card text-primary" aria-label="Select color theme">
            {THEMES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
      </header>
      <main className="grid">
        <section className="bg-card card">
          <h2 className="text-primary">Understanding fake news detection</h2>
          <h3 className="text-primary">{conceptSteps[conceptStep].title}</h3>
          <p className="text-primary">{conceptSteps[conceptStep].content}</p>
          <div className="row">
            <button className="bg-button" onClick={() => setConceptStep(Math.max(0, conceptStep - 1))} disabled={conceptStep === 0}>Previous</button>
            <button className="bg-button" onClick={() => setConceptStep(Math.min(conceptSteps.length - 1, conceptStep + 1))} disabled={conceptStep === conceptSteps.length - 1}>Next</button>
          </div>
        </section>

        <section className="bg-card card">
          <h2 className="text-primary">Results from the last notebook run</h2>
          {error && <p className="text-primary error">Could not load the results ({error}). Serve this folder over HTTP, for example with <code>python -m http.server</code>, instead of opening the file directly.</p>}
          {!results && !error && <p className="text-primary">Loading results…</p>}
          {results && (
            <div>
              <p className="text-primary note">These charts come from a run on synthetic data. They say nothing about how well the approach would work on real news.</p>
              <p className="text-primary">Top selected features ({results.features.length} selected in total):</p>
              <ol className="features text-primary">
                {results.features.slice(0, 20).map(f => <li key={f}>{f}</li>)}
              </ol>
              <Chart figure={results.mi} theme={theme} />
              <Chart figure={results.comparison} theme={theme} />
              {results.metrics && <MetricsTable metrics={results.metrics} />}
            </div>
          )}
        </section>

        <section className="bg-card card">
          <h2 className="text-primary">Dataset overview</h2>
          <p className="text-primary">Synthetic dataset of 20,000 invented news-like articles with about 5% missing values in source and author, and balanced real and fake labels. It was generated by <code>data/generate_synthetic.py</code> with a fixed seed. The original teaching dataset is not in the repository, so this one was regenerated on 2026-10-13.</p>
          <p className="text-primary">Run the notebook yourself: <a href="https://github.com/Janeferdinant/news" className="link">github.com/Janeferdinant/news</a></p>
          <p className="text-primary">This page is a static results view. The Flask + React app that scores pasted text with the three models runs locally; see the README.</p>
        </section>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
