import { render, screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App.jsx';

const MODELS = {
  max_chars: 50,
  notice: 'Trained on synthetic data built for teaching.',
  models: [
    { id: 'logistic_regression', name: 'Logistic Regression', available: true, explains: true, description: 'LR' },
    { id: 'xgboost', name: 'XGBoost', available: true, explains: true, description: 'XGB' },
    { id: 'distilbert', name: 'DistilBERT', available: true, explains: false, description: 'DB' },
  ],
};

const PREDICTION = {
  notice: 'Trained on synthetic data built for teaching.',
  unavailable: [],
  results: [
    {
      model: 'logistic_regression', name: 'Logistic Regression', label: 'fake', confidence: 0.91,
      probabilities: { fake: 0.91, real: 0.09 },
      top_terms: [
        { term: 'shocking', contribution: -1.5, direction: 'fake', value: 0.2 },
        { term: 'officials said', contribution: 0.6, direction: 'real', value: 0.1 },
      ],
    },
    {
      model: 'xgboost', name: 'XGBoost', label: 'real', confidence: 0.77,
      probabilities: { fake: 0.23, real: 0.77 },
      top_terms: [{ term: 'report', contribution: 0.9, direction: 'real', value: 0.3 }],
    },
    {
      model: 'distilbert', name: 'DistilBERT', label: 'fake', confidence: 0.99,
      probabilities: { fake: 0.99, real: 0.01 }, top_terms: null,
    },
  ],
};

function mockFetch(handlers) {
  global.fetch = vi.fn(async (url, options) => {
    const h = handlers[url];
    const out = typeof h === 'function' ? h(options) : h;
    return { ok: out.ok ?? true, status: out.status ?? 200, json: async () => out.body };
  });
}

beforeEach(() => mockFetch({ '/api/models': { body: MODELS }, '/api/predict': { body: PREDICTION } }));
afterEach(() => vi.restoreAllMocks());

describe('App', () => {
  it('shows the synthetic-data banner and no headline metrics', async () => {
    render(<App />);
    expect(screen.getByRole('note')).toHaveTextContent(/synthetic data built for teaching/i);
    expect(screen.getByRole('note')).toHaveTextContent(/not a fact-checking tool/i);
    await screen.findByText(/0 \/ 50/);
    expect(screen.queryByText(/accuracy/i)).not.toBeInTheDocument();
  });

  it('disables Analyse until there is text and enforces the length cap', async () => {
    const user = userEvent.setup();
    render(<App />);
    const button = screen.getByRole('button', { name: /analyse/i });
    await screen.findByText(/0 \/ 50/);
    expect(button).toBeDisabled();
    await user.type(screen.getByLabelText(/article text/i), 'hello');
    expect(button).toBeEnabled();
    expect(screen.getByText(/5 \/ 50/)).toBeInTheDocument();
    await user.clear(screen.getByLabelText(/article text/i));
    await user.click(screen.getByRole('button', { name: /sensational style example/i }));
    expect(screen.getByText(/over the limit/i)).toBeInTheDocument();
    expect(button).toBeDisabled();
  });

  it('posts the article and shows label, confidence and top terms for each model', async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/0 \/ 50/);
    await user.type(screen.getByLabelText(/title/i), 'My title');
    await user.type(screen.getByLabelText(/article text/i), 'Some text');
    await user.click(screen.getByRole('button', { name: /analyse/i }));

    const lr = await screen.findByRole('article', { name: /logistic regression/i });
    expect(within(lr).getByText('Fake')).toBeInTheDocument();
    expect(within(lr).getByText('91%')).toBeInTheDocument();
    expect(within(lr).getByText('shocking')).toBeInTheDocument();
    expect(within(lr).getByText('officials said')).toBeInTheDocument();
    expect(within(lr).getAllByText(/towards fake|towards real/i)).toHaveLength(2);

    const xgb = screen.getByRole('article', { name: /xgboost/i });
    expect(within(xgb).getByText('Real')).toBeInTheDocument();
    expect(within(xgb).getByText('77%')).toBeInTheDocument();

    const bert = screen.getByRole('article', { name: /distilbert/i });
    expect(within(bert).getByText('99%')).toBeInTheDocument();
    expect(within(bert).getByText(/no term-level explanation/i)).toBeInTheDocument();

    const call = global.fetch.mock.calls.find(([url]) => url === '/api/predict');
    expect(JSON.parse(call[1].body)).toEqual({ title: 'My title', text: 'Some text' });
  });

  it('lists a model that is unavailable with its reason', async () => {
    mockFetch({
      '/api/models': { body: MODELS },
      '/api/predict': {
        body: {
          ...PREDICTION,
          results: PREDICTION.results.slice(0, 2),
          unavailable: [{ model: 'distilbert', reason: 'Could not load DistilBERT from distilbert/: OSError' }],
        },
      },
    });
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/0 \/ 50/);
    await user.type(screen.getByLabelText(/article text/i), 'Some text');
    await user.click(screen.getByRole('button', { name: /analyse/i }));
    const bert = await screen.findByRole('article', { name: /distilbert/i });
    expect(within(bert).getByText(/not available/i)).toBeInTheDocument();
    expect(within(bert).getByText(/OSError/)).toBeInTheDocument();
  });

  it('shows the API error message in an alert', async () => {
    mockFetch({
      '/api/models': { body: MODELS },
      '/api/predict': { ok: false, status: 400, body: { error: "'text' must not be empty.", code: 'empty_text' } },
    });
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/0 \/ 50/);
    await user.type(screen.getByLabelText(/article text/i), 'x');
    await user.click(screen.getByRole('button', { name: /analyse/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent("'text' must not be empty.");
  });

  it('explains how to start the API when it is unreachable', async () => {
    global.fetch = vi.fn(async () => {
      throw new TypeError('Failed to fetch');
    });
    render(<App />);
    expect(await screen.findByRole('alert')).toHaveTextContent(/python app\/server\.py/);
  });

  it('fills the form from an example', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole('button', { name: /neutral style example/i }));
    await waitFor(() => expect(screen.getByLabelText(/title/i).value).toMatch(/river clean-up/));
  });
});
