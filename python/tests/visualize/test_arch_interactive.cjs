// Execute the packaged controller with a minimal DOM/Plotly event fixture.
// Browser rendering remains a separate integration check.
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');
const vm = require('node:vm');

const source = readFileSync(path.join(
  __dirname, '../../bloqade/lanes/visualize/_arch_interactive.js'
), 'utf8');

function fixture(occupied) {
  class Element {
    constructor() {
      this.style = {};
      this.attributes = {};
      this.children = [];
      this.isConnected = true;
      this.offsetWidth = 100;
      this.offsetHeight = 40;
    }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener() {}
    appendChild(child) { this.children.push(child); }
    remove() { this.isConnected = false; }
  }
  const overlay = new Element();
  const hover = new Element();
  const events = {};
  const domEvents = {};
  const siteData = [0, 0, 0, 0, 0];
  const atomData = [7, ...siteData, ''];
  const plot = new Element();
  plot.clientWidth = 600;
  plot.clientHeight = 400;
  plot.layout = {font: {color: 'black'}, meta: {
    archVisualizerSiteTraceIndex: 0,
    archVisualizerSiteLanePreviewMode: 'hover',
    archVisualizerSiteLanePathRefs: {'0,0,0': [[0, false]]},
    archVisualizerSiteLanePaths: [{
      exactX: [0, 1], exactY: [0, 0], color: 'red', dash: 'dash'
    }],
    bloqadePlotlyDebugger: {frameNames: ['step-0']}
  }};
  plot.data = [{x: [0], y: [0], customdata: [siteData]}];
  if (occupied) {
    plot.data.push({meta: {bloqadeTraceKind: 'atom'}, customdata: [atomData]});
  }
  plot._fullLayout = {
    xaxis: {_offset: 10, l2p: (x) => 100 * x},
    yaxis: {_offset: 20, l2p: (y) => 100 * y}
  };
  plot.querySelector = (selector) => {
    if (selector === '[data-arch-visualizer-path-overlays]') return overlay;
    if (selector === '.hoverlayer') return hover;
    // Bus panel construction is unrelated to the hover event under test.
    if (selector === '[data-arch-visualizer-bus-selectors]') return new Element();
    return null;
  };
  plot.on = (name, handler) => { (events[name] ??= []).push(handler); };
  plot.addEventListener = (name, handler) => {
    (domEvents[name] ??= []).push(handler);
  };
  plot.getBoundingClientRect = () => ({left: 0, top: 0});
  vm.runInNewContext(source, {
    document: {
      getElementById: () => plot,
      createElement: () => new Element(),
      createElementNS: () => new Element()
    },
    window: {setTimeout: (callback) => callback()}
  });
  return {
    plot, siteData, atomData,
    emit: (name, event) => events[name].forEach((handler) => handler(event)),
    move: (event) => domEvents.mousemove.forEach((handler) => handler(event)),
    leave: () => domEvents.mouseleave.forEach((handler) => handler()),
    lanes: () => overlay.children.filter((element) =>
      element.isConnected && 'data-arch-visualizer-site-lane' in element.attributes
    )
  };
}

test('pointer proximity previews a site even when another overlay captures hover', () => {
  const f = fixture(true);
  f.move({clientX: 10, clientY: 20});
  assert.equal(f.lanes().length, 1);
  f.move({clientX: 300, clientY: 300});
  assert.equal(f.lanes().length, 0);
});

for (const kind of ['empty site', 'occupied site', 'atom']) {
  test(`debugger previews lanes when Plotly hovers an ${kind}`, () => {
    const f = fixture(kind !== 'empty site');
    f.emit('plotly_hover', {points: [{
      curveNumber: kind === 'atom' ? 1 : 0,
      customdata: kind === 'atom' ? f.atomData : f.siteData,
      x: 0, y: 0
    }]});
    assert.equal(f.lanes().length, 1);
    assert.equal(f.lanes()[0].attributes.d, 'M10,20L110,20');
    assert.equal(f.lanes()[0].attributes['stroke-dasharray'], '9,6');
    f.leave();
    assert.equal(f.lanes().length, 0);
  });
}
