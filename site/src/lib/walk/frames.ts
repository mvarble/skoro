import type { GraphSpec } from '$lib/graph/spec';
import { edgeId, type WalkthroughSpec } from './spec';

// What a walkthrough draws after a step: steps `0..=step` folded over the spec. Visibility, kinds,
// annotations, labels and muting carry from step to step; emphasis and flow belong to one step.

export interface Frame {
    step: number;
    /** The nodes, edges and regions drawn. */
    visible: Set<string>;
    muted: Set<string>;
    emphasized: Set<string>;
    flow: Set<string>;
    flowBack: Set<string>;
    /** The figure as it stands, with each node's and edge's kind, annotation and label. */
    graph: GraphSpec;
    /** Region labels as they stand. */
    regionLabels: Map<string, string | undefined>;
}

export function frameAt(spec: WalkthroughSpec, step: number): Frame {
    const last = Math.min(Math.max(step, 0), spec.steps.length - 1);
    const shown = new Set<string>();
    const hiddenEdges = new Set(spec.edges.filter((e) => e.hidden).map(edgeId));
    const muted = new Set<string>([
        ...spec.nodes.filter((n) => n.muted).map((n) => n.id),
        ...spec.edges.filter((e) => e.muted).map(edgeId),
    ]);
    const kinds = new Map<string, string>();
    const annotations = new Map<string, string | null>();
    const labels = new Map<string, string>();
    const edges = new Set(spec.edges.map(edgeId));
    const all = [...spec.nodes.map((n) => n.id), ...(spec.regions ?? []).map((r) => r.id)];

    for (const s of spec.steps.slice(0, last + 1)) {
        for (const id of s.show == 'all' ? [...all, ...edges] : (s.show ?? [])) {
            if (edges.has(id)) hiddenEdges.delete(id);
            else shown.add(id);
        }
        for (const id of s.hide ?? []) {
            if (edges.has(id)) hiddenEdges.add(id);
            else shown.delete(id);
        }
        for (const id of s.mute ?? []) muted.add(id);
        for (const id of s.unmute ?? []) muted.delete(id);
        for (const [id, kind] of Object.entries(s.kind ?? {})) kinds.set(id, kind);
        for (const [id, text] of Object.entries(s.annotate ?? {})) annotations.set(id, text);
        for (const [id, text] of Object.entries(s.label ?? {})) labels.set(id, text);
    }

    const visible = new Set(shown);
    for (const edge of spec.edges) {
        const id = edgeId(edge);
        if (shown.has(edge.from) && shown.has(edge.to) && !hiddenEdges.has(id)) visible.add(id);
    }
    const current = spec.steps[last]!;
    const graph: GraphSpec = {
        ...spec,
        nodes: spec.nodes.map((node) => {
            const annotation = annotations.has(node.id)
                ? annotations.get(node.id)
                : node.annotation;
            return {
                ...node,
                kind: kinds.get(node.id) ?? node.kind,
                label: labels.get(node.id) ?? node.label,
                annotation: annotation ?? undefined,
            };
        }),
        edges: spec.edges.map((edge) => ({ ...edge, kind: kinds.get(edgeId(edge)) ?? edge.kind })),
    };
    return {
        step: last,
        visible,
        muted,
        emphasized: new Set(current.emphasize ?? []),
        flow: new Set(current.flow ?? []),
        flowBack: new Set(current.flowBack ?? []),
        graph,
        regionLabels: new Map((spec.regions ?? []).map((r) => [r.id, labels.get(r.id) ?? r.label])),
    };
}

/** Every kind of node and edge some step of the walkthrough draws, for its legend. */
export function kindsUsed(spec: WalkthroughSpec): { nodes: Set<string>; edges: Set<string> } {
    const nodeIds = new Set(spec.nodes.map((n) => n.id));
    const nodes = new Set(spec.nodes.map((n) => n.kind));
    const edges = new Set(spec.edges.flatMap((e) => (e.kind === undefined ? [] : [e.kind])));
    for (const step of spec.steps) {
        for (const [id, kind] of Object.entries(step.kind ?? {})) {
            (nodeIds.has(id) ? nodes : edges).add(kind);
        }
    }
    return { nodes, edges };
}
