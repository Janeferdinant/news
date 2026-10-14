import React from 'react';

const pct = (p) => `${Math.round(p * 100)}%`;

function Terms({ terms }) {
  const max = Math.max(...terms.map((t) => Math.abs(t.contribution)), 1e-9);
  return (
    <ul className="terms" aria-label="Top contributing terms">
      {terms.map((t) => (
        <li key={t.term}>
          <span className="term">{t.term}</span>
          <span className={`pill ${t.direction}`}>{t.direction === 'fake' ? 'towards fake' : 'towards real'}</span>
          <span className="bar" aria-hidden="true">
            <span className={`bar-fill ${t.direction}`} style={{ width: `${(100 * Math.abs(t.contribution)) / max}%` }} />
          </span>
        </li>
      ))}
    </ul>
  );
}

export default function ModelCard({ result, unavailable, name }) {
  const heading = result ? result.name : name;
  const id = `card-${result ? result.model : name.replace(/\s+/g, '-').toLowerCase()}`;

  if (!result) {
    return (
      <article className="card unavailable" aria-labelledby={id}>
        <h3 id={id}>{heading}</h3>
        <p className="muted"><strong>Not available.</strong> {unavailable}</p>
      </article>
    );
  }

  return (
    <article className="card" aria-labelledby={id}>
      <h3 id={id}>{heading}</h3>
      <p className="verdict">
        <span className={`label ${result.label}`}>{result.label === 'fake' ? 'Fake' : 'Real'}</span>
        <span className="confidence">{pct(result.confidence)}</span>
      </p>
      <div className="meter" role="img" aria-label={`Confidence ${pct(result.confidence)}`}>
        <span className={`meter-fill ${result.label}`} style={{ width: pct(result.confidence) }} />
      </div>
      <p className="muted small">
        Confidence is the model's probability for its own label (fake {pct(result.probabilities.fake)}, real{' '}
        {pct(result.probabilities.real)}).
      </p>
      {result.top_terms && result.top_terms.length > 0 && (
        <>
          <h4>Top contributing terms</h4>
          <Terms terms={result.top_terms} />
          {result.selected_features_total !== undefined && (
            <p className="muted small">
              {result.selected_features_present} of the {result.selected_features_total} features chosen by ant-colony
              selection appear in this article.
            </p>
          )}
        </>
      )}
      {result.top_terms && result.top_terms.length === 0 && (
        <p className="muted small">None of the selected features appear in this article, so there are no terms to show.</p>
      )}
      {result.top_terms === null && (
        <p className="muted small">{result.note || 'No term-level explanation for this model.'}</p>
      )}
    </article>
  );
}
