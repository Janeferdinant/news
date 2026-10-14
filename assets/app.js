// Generated from assets/app.jsx by scripts/build_jsx.mjs. Do not edit.
// Source of assets/app.js. Rebuild with: node scripts/build_jsx.mjs (see README).
// Reads the files written by the notebook (outputs/*.json, outputs/selected_features.txt).
const {
  useState,
  useEffect,
  useRef
} = React;
const THEMES = [['modern', 'Modern'], ['git', 'Git Dark'], ['rainbow', 'Rainbow Gradient'], ['neon', 'Neon'], ['pastel', 'Pastel'], ['monochrome', 'Monochrome'], ['cyberpunk', 'Cyberpunk'], ['retro', 'Retro']];
function Chart({
  figure,
  theme
}) {
  const ref = useRef(null);
  useEffect(() => {
    if (!figure || !ref.current) return;
    const color = getComputedStyle(ref.current).color;
    const layout = Object.assign({}, figure.layout, {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: {
        color
      },
      autosize: true,
      margin: {
        l: 50,
        r: 10,
        t: 70,
        b: 90
      }
    });
    // Plotly does not wrap titles, so break long ones before the parenthesis and left-align them
    if (layout.title && layout.title.text) {
      layout.title = Object.assign({}, layout.title, {
        text: layout.title.text.replace(' (', '<br>('),
        x: 0.02,
        xanchor: 'left',
        font: {
          size: 15
        }
      });
    }
    delete layout.template;
    Plotly.react(ref.current, figure.data, layout, {
      responsive: true,
      displayModeBar: false
    });
  }, [figure, theme]);
  return /*#__PURE__*/React.createElement("div", {
    ref: ref,
    className: "chart",
    role: "img",
    "aria-label": figure && figure.layout.title ? figure.layout.title.text : 'chart'
  });
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
    Promise.all([get('outputs/mi_scores.json'), get('outputs/model_comparison.json'), get('outputs/selected_features.txt', 'text')]).then(([mi, comparison, features]) => setResults({
      mi,
      comparison,
      features: features.split(/\r?\n/).filter(Boolean)
    })).catch(err => setError(String(err.message || err)));
  }, []);
  const conceptSteps = [{
    title: 'What is fake news detection?',
    content: 'Fake news detection tries to label articles as real or fake from their text and metadata using ML and NLP. In this demo the articles are synthetic, so the labels come from a writing-style recipe, not from checking facts.'
  }, {
    title: 'Ant Colony Optimization (ACO)',
    content: 'ACO mimics ant foraging to pick useful features. Each "ant" samples a set of features, favouring those with high mutual information and strong pheromone, and features in well-scoring sets get more pheromone. Scores come from a validation split of the training data only.'
  }, {
    title: 'Clustering-based mutual information',
    content: 'Each feature\'s values are grouped into 3 clusters, and mutual information measures how much the cluster tells you about the label. It is a quick filter that can pick up non-linear patterns.'
  }, {
    title: 'Embedded-filter ACO',
    content: 'It combines the speed of the filter (clustering-based MI) with the search of a wrapper (ACO). The search is limited to 50 to 200 features per ant.'
  }];
  return /*#__PURE__*/React.createElement("div", {
    className: theme + '-theme page'
  }, /*#__PURE__*/React.createElement("div", {
    className: "banner",
    role: "note"
  }, "Trained on synthetic data built for teaching; not a fact-checking tool."), /*#__PURE__*/React.createElement("header", {
    className: "bar"
  }, /*#__PURE__*/React.createElement("h1", {
    className: "text-primary"
  }, "Fake News Detection Explorer"), /*#__PURE__*/React.createElement("label", {
    className: "theme-picker text-primary"
  }, /*#__PURE__*/React.createElement("span", null, "Theme"), /*#__PURE__*/React.createElement("select", {
    value: theme,
    onChange: e => setTheme(e.target.value),
    className: "bg-card text-primary",
    "aria-label": "Select color theme"
  }, THEMES.map(([value, label]) => /*#__PURE__*/React.createElement("option", {
    key: value,
    value: value
  }, label))))), /*#__PURE__*/React.createElement("main", {
    className: "grid"
  }, /*#__PURE__*/React.createElement("section", {
    className: "bg-card card"
  }, /*#__PURE__*/React.createElement("h2", {
    className: "text-primary"
  }, "Understanding fake news detection"), /*#__PURE__*/React.createElement("h3", {
    className: "text-primary"
  }, conceptSteps[conceptStep].title), /*#__PURE__*/React.createElement("p", {
    className: "text-primary"
  }, conceptSteps[conceptStep].content), /*#__PURE__*/React.createElement("div", {
    className: "row"
  }, /*#__PURE__*/React.createElement("button", {
    className: "bg-button",
    onClick: () => setConceptStep(Math.max(0, conceptStep - 1)),
    disabled: conceptStep === 0
  }, "Previous"), /*#__PURE__*/React.createElement("button", {
    className: "bg-button",
    onClick: () => setConceptStep(Math.min(conceptSteps.length - 1, conceptStep + 1)),
    disabled: conceptStep === conceptSteps.length - 1
  }, "Next"))), /*#__PURE__*/React.createElement("section", {
    className: "bg-card card"
  }, /*#__PURE__*/React.createElement("h2", {
    className: "text-primary"
  }, "Results from the last notebook run"), error && /*#__PURE__*/React.createElement("p", {
    className: "text-primary error"
  }, "Could not load the results (", error, "). Serve this folder over HTTP, for example with ", /*#__PURE__*/React.createElement("code", null, "python -m http.server"), ", instead of opening the file directly."), !results && !error && /*#__PURE__*/React.createElement("p", {
    className: "text-primary"
  }, "Loading results\u2026"), results && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "text-primary note"
  }, "These charts come from a run on synthetic data. They say nothing about how well the approach would work on real news."), /*#__PURE__*/React.createElement("p", {
    className: "text-primary"
  }, "Top selected features (", results.features.length, " selected in total):"), /*#__PURE__*/React.createElement("ol", {
    className: "features text-primary"
  }, results.features.slice(0, 20).map(f => /*#__PURE__*/React.createElement("li", {
    key: f
  }, f))), /*#__PURE__*/React.createElement(Chart, {
    figure: results.mi,
    theme: theme
  }), /*#__PURE__*/React.createElement(Chart, {
    figure: results.comparison,
    theme: theme
  }))), /*#__PURE__*/React.createElement("section", {
    className: "bg-card card"
  }, /*#__PURE__*/React.createElement("h2", {
    className: "text-primary"
  }, "Dataset overview"), /*#__PURE__*/React.createElement("p", {
    className: "text-primary"
  }, "Synthetic dataset of 20,000 invented news-like articles with about 5% missing values in source and author, and balanced real and fake labels. It was generated by ", /*#__PURE__*/React.createElement("code", null, "data/generate_synthetic.py"), " with a fixed seed. The original teaching dataset is not in the repository, so this one was regenerated on 2026-10-13."), /*#__PURE__*/React.createElement("p", {
    className: "text-primary"
  }, "Run the notebook yourself: ", /*#__PURE__*/React.createElement("a", {
    href: "https://github.com/Janeferdinant/news",
    className: "link"
  }, "github.com/Janeferdinant/news")))));
}
ReactDOM.createRoot(document.getElementById('root')).render( /*#__PURE__*/React.createElement(App, null));
