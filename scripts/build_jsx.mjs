// Compiles assets/app.jsx to assets/app.js, so the page needs no in-browser Babel.
//   npm install --no-save @babel/standalone@7.22.9
//   node scripts/build_jsx.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { transform } from '@babel/standalone';

const source = readFileSync(new URL('../assets/app.jsx', import.meta.url), 'utf8');
const { code } = transform(source, { presets: ['react'], filename: 'app.jsx' });
writeFileSync(new URL('../assets/app.js', import.meta.url), `// Generated from assets/app.jsx by scripts/build_jsx.mjs. Do not edit.\n${code}\n`);
console.log('wrote assets/app.js');
