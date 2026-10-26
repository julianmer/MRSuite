// The canvas: blocks wired together, drawn with Vue Flow. Python holds the graph; this draws
// what it is sent and says what the user did.
import { Handle, Position, VueFlow, useVueFlow } from "@vue-flow/core";
import { Background } from "@vue-flow/background";

// Vue Flow's files are built for a bundler, which writes NODE_ENV in; without one, its warnings
// (and its check for its styles at every mount) throw on the missing process
globalThis.process ??= { env: { NODE_ENV: "production" } };

// one block: its ports, a small picture of what it made with a line on it, its buttons, and
// a card on hover saying what it does and how
const Block = {
  props: ["id", "data", "selected"],
  components: { Handle },
  inject: ["act"],
  setup() {
    return { Position };
  },
  methods: {
    line(y) {
      const step = 120 / Math.max(y.length - 1, 1);
      return y.map((v, i) => `${(i * step).toFixed(1)},${(34 - v * 32).toFixed(1)}`).join(" ");
    },
    bar(y, i) {
      const w = 120 / y.length;
      return { x: i * w + 1, width: Math.max(w - 2, 1), y: 34 - y[i] * 32, height: y[i] * 32 };
    },
  },
  template: `
    <div class="mr-block" :class="['is-' + data.status, { 'is-selected': selected }]"
         :style="{ '--accent': data.colour }" :title="data.detail || ''">
      <div class="mr-block-head">
        <span class="mr-status"></span>
        <span class="mr-block-title">{{ data.title }}</span>
        <span class="mr-block-name">{{ data.name }}</span>
      </div>
      <div class="mr-block-tool" v-if="data.tool">{{ data.tool }}</div>
      <div class="mr-ports">
        <div class="mr-port mr-in" v-for="p in data.inputs" :key="'i' + p.name">
          <Handle type="target" :position="Position.Left" :id="p.name"
                  :class="'mr-kind-' + p.kind" />
          <span>{{ p.name }}{{ p.many ? ' …' : '' }}</span>
          <span class="mr-value" v-if="p.value" :title="p.value">{{ p.value }}</span>
        </div>
        <div class="mr-port mr-out" v-for="p in data.outputs" :key="'o' + p.name">
          <span>{{ p.name }}</span>
          <Handle type="source" :position="Position.Right" :id="p.name"
                  :class="'mr-kind-' + p.kind" />
        </div>
      </div>
      <div v-if="data.thumb" class="mr-thumb-box" :class="'mr-kind-' + data.thumb.kind">
        <svg v-if="data.thumb.kind === 'spectra'" class="mr-thumb"
             viewBox="0 0 120 36" preserveAspectRatio="none">
          <polyline v-for="(y, i) in data.thumb.faint" :key="i" class="mr-faint" :points="line(y)" />
          <polyline v-if="data.thumb.line" :points="line(data.thumb.line)" />
        </svg>
        <svg v-else-if="data.thumb.kind === 'fit'" class="mr-thumb mr-fit"
             viewBox="0 0 120 36" preserveAspectRatio="none">
          <polyline class="mr-data" :points="line(data.thumb.data)" />
          <polyline class="mr-baseline" :points="line(data.thumb.baseline)" />
          <polyline :points="line(data.thumb.fit)" />
          <polyline class="mr-residual" :points="line(data.thumb.residual)" />
        </svg>
        <svg v-else-if="data.thumb.kind === 'bars'" class="mr-thumb"
             viewBox="0 0 120 36" preserveAspectRatio="none">
          <rect v-for="(v, i) in data.thumb.y" :key="i" v-bind="bar(data.thumb.y, i)" />
        </svg>
        <img v-else-if="data.thumb.kind === 'image'" class="mr-thumb-image" :src="data.thumb.src" />
        <div v-else class="mr-thumb-text">{{ data.thumb.text }}</div>
        <div v-if="data.thumb.labels && data.thumb.labels.length" class="mr-thumb-names">
          <span v-for="(l, i) in data.thumb.labels" :key="i">{{ l }}</span>
        </div>
        <div v-if="data.thumb.label" class="mr-thumb-label">{{ data.thumb.label }}</div>
      </div>
      <div v-if="data.about && data.about.length" class="mr-card nodrag">
        <div class="mr-card-help">{{ data.help }}</div>
        <div v-for="(row, i) in data.about" :key="i" class="mr-card-row">
          <span class="mr-card-key">{{ row[0] }}</span><span class="mr-card-value">{{ row[1] }}</span>
        </div>
      </div>
      <div v-if="data.error" class="mr-error">{{ data.error }}</div>
      <div class="mr-actions nodrag">
        <button title="Run up to here" @click.stop="act(id, 'run')">
          <i class="material-icons">play_arrow</i></button>
        <button title="Preview what it made" @click.stop="act(id, 'preview')">
          <i class="material-icons">visibility</i></button>
        <button title="Header and history: what it made, as NIfTI-MRS records it"
                @click.stop="act(id, 'header')">
          <i class="material-icons">receipt_long</i></button>
        <button title="Export what it made" @click.stop="act(id, 'export')">
          <i class="material-icons">download</i></button>
        <button title="Remove" @click.stop="act(id, 'remove')">
          <i class="material-icons">close</i></button>
      </div>
    </div>`,
};

export default {
  components: { VueFlow, Background, Block },
  props: { nodes: Array, edges: Array },
  provide() {
    return { act: (id, what) => this.$emit("act", { id, what }) };
  },
  setup() {
    return { flow: useVueFlow("mrsuite") };
  },
  data() {
    return { pending: null, joined: false };
  },
  methods: {
    at(event) {
      return this.flow.screenToFlowCoordinate({ x: event.clientX, y: event.clientY });
    },
    port(id, name, side) {
      const node = this.nodes.find((n) => n.id === id);
      return node && node.data[side].find((p) => p.name === name);
    },
    valid(c) {
      const out = this.port(c.source, c.sourceHandle, "outputs");
      const into = this.port(c.target, c.targetHandle, "inputs");
      return !!out && !!into && c.source !== c.target &&
        (out.kind === "any" || into.kind === "any" || out.kind === into.kind);
    },
    connect(c) {
      this.joined = true;
      this.$emit("connect", c);
    },
    start(e) {
      this.joined = false;
      this.pending = e;
    },
    end(event) {
      // a wire let go over empty canvas: offer the blocks it can go into
      const over = this.empty(event);
      if (!this.joined && this.pending && over) {
        const point = event.changedTouches ? event.changedTouches[0] : event;
        this.$emit("loose", { ...this.pending, ...this.at(point) });
      }
      this.pending = null;
    },
    dragged(e) {
      this.$emit("moved", e.nodes.map((n) => ({ id: n.id, x: n.position.x, y: n.position.y })));
    },
    empty(event) {
      // the canvas itself, not a block or a wire on it
      const t = event && event.target;
      return !!(t && t.closest && t.closest(".vue-flow__pane") &&
        !t.closest(".vue-flow__node, .vue-flow__edge"));
    },
    clicked(event) {
      if (this.empty(event)) this.$emit("search", this.at(event));
    },
    keyed(event) {
      // only while the canvas has the focus: a key typed elsewhere is not for it
      if (event.key !== "Delete" && event.key !== "Backspace") return;
      event.preventDefault();                       // not the browser's own Back
      const nodes = this.flow.getSelectedNodes.value.map((n) => n.id);
      const edges = this.flow.getSelectedEdges.value.map((e) => e.id);
      if (nodes.length) this.$emit("removed", { kind: "node", ids: nodes });
      if (edges.length) this.$emit("removed", { kind: "edge", ids: edges });
    },
    dropped(event) {
      const key = event.dataTransfer.getData("text/mrsuite-block");
      if (key) this.$emit("dropped", { key, ...this.at(event) });
    },
    fit(padding = 0.25) {
      this.flow.fitView({ padding, maxZoom: 1.1, duration: 300 });
    },
  },
  template: `
    <div class="mr-canvas" tabindex="0" @dblclick="clicked" @keydown="keyed"
         @dragover.prevent @drop.prevent="dropped">
      <VueFlow id="mrsuite" :nodes="nodes" :edges="edges" :is-valid-connection="valid"
               :zoom-on-double-click="false" :delete-key-code="null"
               :min-zoom="0.2" :max-zoom="2" :snap-to-grid="true" :snap-grid="[8, 8]"
               @connect="connect" @connect-start="start" @connect-end="end"
               @node-drag-stop="dragged" @node-click="$emit('select', { id: $event.node.id })"
               @pane-click="$emit('select', { id: null })">
        <template #node-block="props"><Block v-bind="props" /></template>
        <Background :gap="20" :size="1.2" />
      </VueFlow>
    </div>`,
};
