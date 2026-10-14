// Thin client for the Flask API. Errors carry the server's message when there is one.
async function request(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch {
    throw new Error('Cannot reach the API. Start it with: python app/server.py');
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON error page */
  }
  if (!res.ok) {
    throw new Error((body && body.error) || `The API answered with HTTP ${res.status}.`);
  }
  return body;
}

export const fetchModels = () => request('/api/models');

export const predict = ({ title, text }) =>
  request('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, text }),
  });
