import type { EdgeSpec, GraphSpec, Hue, NodeSpec } from '$lib/graph/spec';

// The input to `<Walkthrough>`: a graph figure that is built up step by step, beside the code that
// builds it. Every node, edge and region sits where it ends up, in the coordinates of
// `$lib/graph/spec.ts`, and each step says what changes: what appears or goes, which kind a node
// or edge becomes, what is muted, what is drawn attention to, and which edges carry a pass.
//
// Nodes and regions start hidden, and a step `show`s them. An edge is drawn when both its nodes
// are, unless a step `hide`s it (or it starts `hidden`), so building a node brings its input
// edges with it.
//
// A step points at the code it is about by text, not by line number: each `code` match is a
// substring of a line, or a `from`/`to` pair spanning the lines from one to the other. The quoted
// code is the page's own file-reference fence, and the `.md` export fails the build when a match
// no longer finds a line in it, so a walkthrough cannot drift from the code it quotes.

/** A labelled box drawn behind the graph: a scope, a snapshot, a cache. */
export interface RegionSpec {
    id: string;
    /** The bottom-left corner, in mathematical coordinates. */
    x: number;
    y: number;
    width: number;
    height: number;
    /** Shown inside its top-left corner; may contain `$…$` TeX. */
    label?: string;
    /** Defaults to `gray`. */
    hue?: Hue;
    /** Defaults to `solid`. */
    line?: 'solid' | 'dashed';
}

export interface WalkEdgeSpec extends EdgeSpec {
    /** How steps name the edge. Defaults to `from->to`. */
    id?: string;
    /** Not drawn until a step shows it, even once both its nodes are. */
    hidden?: boolean;
}

/** A line of the quoted code containing `text`, or the lines from one containing `from` to the next containing `to`. */
export type CodeMatch = string | { from: string; to: string };

export interface StepSpec {
    /** What happens, in a sentence or two. Like every text, it may contain `$…$` and `` `code` ``. */
    note: string;
    /** The lines of the quoted code this step is about. */
    code?: CodeMatch | CodeMatch[];
    /** Nodes, edges and regions that appear, or `all` of them. */
    show?: string[] | 'all';
    /** Nodes, edges and regions that go. */
    hide?: string[];
    /** Nodes and edges that change kind, to a key of `nodeKinds` or `edgeKinds`. */
    kind?: Record<string, string>;
    /** Nodes whose annotation changes; `null` removes it. */
    annotate?: Record<string, string | null>;
    /** Nodes and regions whose label changes. */
    label?: Record<string, string>;
    /** Nodes, edges and regions drawn faintly from this step on, or no longer. */
    mute?: string[];
    unmute?: string[];
    /** Nodes, edges and regions drawn attention to, in this step alone. */
    emphasize?: string[];
    /** Edges a pass runs along (`flow`) or against (`flowBack`), in this step alone. */
    flow?: string[];
    flowBack?: string[];
}

export interface WalkthroughSpec extends Omit<GraphSpec, 'nodes' | 'edges'> {
    nodes: NodeSpec[];
    edges: WalkEdgeSpec[];
    regions?: RegionSpec[];
    steps: StepSpec[];
}

export const edgeId = (edge: WalkEdgeSpec) => edge.id ?? `${edge.from}->${edge.to}`;

/** The matches of a step's `code`, as a list. */
export const codeMatches = (step: StepSpec): CodeMatch[] =>
    step.code === undefined ? [] : Array.isArray(step.code) ? step.code : [step.code];

/**
 * The indices of the `lines` a match picks out: every line containing a string match, or each
 * span from a line containing `from` to the next line, at or after it, containing `to`.
 */
export function matchLines(lines: string[], match: CodeMatch): number[] {
    if (typeof match == 'string') {
        return lines.flatMap((line, i) => (line.includes(match) ? [i] : []));
    }
    const found: number[] = [];
    for (let i = 0; i < lines.length; i++) {
        if (!lines[i]!.includes(match.from)) continue;
        const end = lines.findIndex((line, j) => j >= i && line.includes(match.to));
        if (end < 0) break;
        for (let j = i; j <= end; j++) found.push(j);
        i = end;
    }
    return found;
}

/** A match as text, for messages and the Markdown export. */
export const showMatch = (match: CodeMatch) =>
    typeof match == 'string' ? match : `${match.from} … ${match.to}`;

/** Checks the spec and returns it: a mistake fails the build, naming the walkthrough. */
export function defineWalkthrough(spec: WalkthroughSpec): WalkthroughSpec {
    const fail = (message: string): never => {
        throw new Error(`Walkthrough "${spec.title}": ${message}`);
    };
    const nodes = new Set<string>();
    const edges = new Set<string>();
    const regions = new Set<string>();
    const claim = (id: string, set: Set<string>) => {
        if (nodes.has(id) || edges.has(id) || regions.has(id)) fail(`"${id}" names two things.`);
        set.add(id);
    };
    for (const node of spec.nodes) {
        claim(node.id, nodes);
        if (!spec.nodeKinds[node.kind])
            fail(`node "${node.id}" is of an undefined kind "${node.kind}".`);
    }
    for (const edge of spec.edges) {
        claim(edgeId(edge), edges);
        for (const end of [edge.from, edge.to]) {
            if (!nodes.has(end)) fail(`edge "${edgeId(edge)}" names no node "${end}".`);
        }
        if (edge.kind !== undefined && !spec.edgeKinds?.[edge.kind]) {
            fail(`edge "${edgeId(edge)}" is of an undefined kind "${edge.kind}".`);
        }
    }
    for (const region of spec.regions ?? []) claim(region.id, regions);
    if (!spec.steps.length) fail('it has no steps.');

    const any = (id: string) => nodes.has(id) || edges.has(id) || regions.has(id);
    spec.steps.forEach((step, i) => {
        const at = `step ${i + 1}`;
        if (!step.note.trim()) fail(`${at} has no note.`);
        const check = (
            field: string,
            ids: Iterable<string>,
            ok: (id: string) => boolean,
            what: string,
        ) => {
            for (const id of ids)
                if (!ok(id)) fail(`${at}: \`${field}\` names "${id}", which is no ${what}.`);
        };
        if (step.show !== 'all') check('show', step.show ?? [], any, 'node, edge or region');
        check('hide', step.hide ?? [], any, 'node, edge or region');
        check('mute', step.mute ?? [], any, 'node, edge or region');
        check('unmute', step.unmute ?? [], any, 'node, edge or region');
        check('emphasize', step.emphasize ?? [], any, 'node, edge or region');
        check('flow', step.flow ?? [], (id) => edges.has(id), 'edge');
        check('flowBack', step.flowBack ?? [], (id) => edges.has(id), 'edge');
        check('annotate', Object.keys(step.annotate ?? {}), (id) => nodes.has(id), 'node');
        check(
            'label',
            Object.keys(step.label ?? {}),
            (id) => nodes.has(id) || regions.has(id),
            'node or region',
        );
        for (const [id, kind] of Object.entries(step.kind ?? {})) {
            if (nodes.has(id)) {
                if (!spec.nodeKinds[kind])
                    fail(`${at}: node "${id}" becomes an undefined kind "${kind}".`);
            } else if (edges.has(id)) {
                if (!spec.edgeKinds?.[kind])
                    fail(`${at}: edge "${id}" becomes an undefined kind "${kind}".`);
            } else {
                fail(`${at}: \`kind\` names "${id}", which is no node or edge.`);
            }
        }
    });
    return spec;
}
