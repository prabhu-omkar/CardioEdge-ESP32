// Runs the dashboard script against a fake device, with no browser or hardware.
//   node tests/dashboard_check.js
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '../firmware/web/dashboard.html'), 'utf8');
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];

const els = {};
function el(id) {
  if (!els[id]) {
    els[id] = {
      id, textContent: '', className: '', style: {}, innerHTML: '', value: '', children: [], clientWidth: 600, clientHeight: 170,
      querySelector() { return (this.b = this.b || { textContent: '' }); },
      append(c) { this.children.push(c); },
      getContext: () => new Proxy({}, { get: () => () => {}, set: () => true }),
    };
  }
  return els[id];
}
for (const m of html.matchAll(/id="(\w+)"/g)) el(m[1]);

global.document = { getElementById: el, createElement: () => ({ value: '', textContent: '' }) };
global.location = { protocol: 'http:', host: '192.168.4.1' };
global.devicePixelRatio = 1;
global.addEventListener = () => {};
let frames = 0;
global.requestAnimationFrame = (f) => { if (frames++ < 3) setImmediate(f); };

let socket;
global.WebSocket = class { constructor(url) { this.url = url; socket = this; } send(v) { this.sent = v; } };
eval(script);

function assert(cond, msg) { if (!cond) { console.error('FAIL', msg); process.exit(1); } }
const base = { src: 1, srcName: 'Demo AFib', nsrc: 7, hr: 96.2, sdnn: 80, rmssd: 90, pnn50: 40, rhythm: 'AFib', rhythmRf: 'AFib',
  beat: 'AFib', beatRf: 'Normal', conf: 0.91, pvc: 0, alert: 1, beats: 100, agree: 93, cnnUs: 42000, rfUs: 180, good: 1, off: 0,
  ampOk: 1, snrOk: 1, baseOk: 1, morphOk: 1, amp: 1.2, snr: 24.1, base: 0.1, morph: 1 };

assert(socket.url === 'ws://192.168.4.1/ws', 'websocket url');
socket.onopen();
assert(el('conn').textContent === 'Connected', 'connected');

socket.onmessage({ data: JSON.stringify(base) });
assert(el('rhythm').textContent === 'AFib', 'rhythm shown');
assert(el('hr').textContent === '96', 'heart rate');
assert(el('banner').className === 'alert', 'alert banner');
assert(el('tCnn').textContent === '42.0 ms', 'inference time');
assert(el('agree').textContent.includes('93.0%'), 'agreement');

socket.onmessage({ data: JSON.stringify({ ...base, conf: -1 }) });
assert(el('bCnn').textContent === 'AFib (rhythm rule)', 'rule decision has no misleading probability');

socket.onmessage({ data: new Int16Array([100, 200, -50]).buffer });

socket.onmessage({ data: JSON.stringify({ ...base, good: 0, alert: 0 }) });
assert(el('banner').textContent.includes('reposition electrodes'), 'poor signal text');
assert(el('rhythm').textContent === 'Paused', 'verdict paused');

socket.onmessage({ data: JSON.stringify({ ...base, good: 0, alert: 0, off: 1 }) });
assert(el('banner').textContent.includes('Leads off'), 'leads off text');

socket.onmessage({ data: JSON.stringify({ ...base, rhythm: '', rhythmRf: '', beat: '', beatRf: '', beats: 0, alert: 0, hr: 0, sdnn: 0 }) });
assert(el('rhythm').textContent === 'Analysing…', 'analysing state');
assert(el('hr').textContent === '—', 'empty heart rate');

el('src').onchange({ target: { value: '3' } });
assert(socket.sent === '3', 'source selection sent');

setTimeout(() => { console.log('dashboard OK'); process.exit(0); }, 20);
