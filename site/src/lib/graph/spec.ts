// The input to `<ModelGraph>`: a hand-placed diagram of a model's graph.
//
// Every piece of text (node labels, annotations, edge labels) may contain inline KaTeX between
// dollar signs, e.g. String.raw`$\log p(x \mid \mu)$`. `\$` is a literal dollar sign.
//
// Coordinates are mathematical, not SVG: the origin is the bottom-left corner of the viewport, +X
// points right and +Y points **up**. Angles are in degrees, counter-clockwise from +X, so 90° is
// straight up. Lengths are in viewport units; the figure scales to fit its column.

/** The variants of `skoro_graph::node::Node`, plus `given` for a cut boundary. */
export type NodeKind =
    'constant' | 'parameter' | 'stochastic' | 'observed' | 'mapped' | 'log-likelihood' | 'given';

/** The binding roles of `skoro_model::roles`, spelled as their `Role` strings. */
export type Role =
    'argument' | 'residual' | 'location' | 'whitening' | 'normalizer_parent' | 'fisher_root';

/**
 * - `parent`: a DAG edge, drawn from the parent to the child (the generative direction).
 * - `scores`: from a log-likelihood term to the stochastic quantity it is the density of.
 * - `binding`: from a term to the node that plays `role` for it.
 * - `plain`: a neutral connector, for annotation.
 */
export type EdgeKind = 'parent' | 'scores' | 'binding' | 'plain';

export type Point = [x: number, y: number];

export interface NodeSpec {
    id: string;
    /** The center, in mathematical coordinates. */
    x: number;
    y: number;
    kind: NodeKind;
    /** Radius, or half the side of a square. Defaults to the graph's `nodeRadius`. */
    r?: number;
    /** Defaults to `square` for a log-likelihood term (the factor-graph convention), else `circle`. */
    shape?: 'circle' | 'square';
    /** Text drawn inside the node: plain (`x₁`) or TeX (`$x_1$`). */
    label?: string;
    /** Text drawn just outside the node, e.g. a quantity type or an operation. */
    annotation?: string;
    /** Where the annotation sits, as an angle on the node. Defaults to -90 (below). */
    annotationAngle?: number;
    /** A hover tooltip. */
    title?: string;
    /** Draw faintly, to focus attention elsewhere. */
    muted?: boolean;
    class?: string;
    style?: string;
}

export interface EdgeSpec {
    from: string;
    to: string;
    /** Defaults to `parent`. */
    kind?: EdgeKind;
    /** Required when `kind` is `binding`, where it picks the colour. */
    role?: Role;
    /**
     * Where the edge leaves `from` / meets `to`, as an angle on that node. Defaults to the direction
     * of the edge's next point (the other node's center, or the nearest `via` point).
     */
    fromAngle?: number;
    toAngle?: number;
    /** Intermediate points the edge passes through. */
    via?: Point[];
    /**
     * `straight` joins the points with line segments. `bezier` draws a smooth curve through them
     * that leaves and meets each node along its normal. Defaults to `bezier` when `bend` is set.
     */
    curve?: 'straight' | 'bezier';
    /**
     * Rotate both default endpoint angles by this many degrees, as TikZ's `bend left` does:
     * positive bows the edge to the left of its direction of travel. Ignored when `via` is set.
     */
    bend?: number;
    /** Defaults to `end`. */
    arrow?: 'end' | 'start' | 'both' | 'none';
    label?: string;
    /** How far along the edge the label sits, from 0 to 1. Defaults to 0.5. */
    labelAt?: number;
    muted?: boolean;
    class?: string;
    style?: string;
}

export interface GraphSpec {
    /** The viewport. */
    width: number;
    height: number;
    /** The figure's accessible name, also shown as the start of its caption. */
    title: string;
    /** A sentence or two explaining the figure. Kept in the Markdown export. */
    caption?: string;
    nodes: NodeSpec[];
    edges: EdgeSpec[];
    /** Default node radius. Defaults to 22. */
    nodeRadius?: number;
    /** Text size, in viewport units. Defaults to 15. */
    fontSize?: number;
    /** Arrowhead length, in viewport units. Defaults to 10. */
    arrowSize?: number;
    /** KaTeX macros for this figure, over the site-wide ones in `katex.config.ts`. */
    katexMacros?: Record<string, string>;
    /** Show a legend of the node and edge kinds used. Defaults to true. */
    legend?: boolean;
}

/** An identity function that type-checks a spec where it is written. */
export const defineGraph = (spec: GraphSpec): GraphSpec => spec;

export const NODE_KIND_NAMES: Record<NodeKind, string> = {
    constant: 'Constant',
    parameter: 'Parameter',
    stochastic: 'Stochastic',
    observed: 'Observed',
    mapped: 'Mapped',
    'log-likelihood': 'LogLikelihood',
    given: 'Given (cut)',
};

export const ROLE_NAMES: Record<Role, string> = {
    argument: 'ARGUMENT',
    residual: 'RESIDUAL',
    location: 'LOCATION',
    whitening: 'WHITENING',
    normalizer_parent: 'NORMALIZER_PARENT',
    fisher_root: 'FISHER_ROOT',
};

export const EDGE_KIND_NAMES: Record<Exclude<EdgeKind, 'binding'>, string> = {
    parent: 'parent',
    scores: 'scores',
    plain: 'connector',
};
