import { hasTex } from '$lib/tex';
import { EDGE_KIND_NAMES, NODE_KIND_NAMES, ROLE_NAMES, type GraphSpec } from './spec';

// A graph figure as Markdown text, for the `.md` exports: a reader without the SVG still gets every node
// and edge, and which kind each one is.
export function describeGraph(spec: GraphSpec): string {
    // A label with math is left as `$…$`, which Markdown readers treat as math; anything else is
    // set as code.
    const quote = (text: string) => (hasTex(text) ? text : `\`${text}\``);
    const name = new Map(spec.nodes.map((n) => [n.id, quote(n.label ?? n.id)]));
    const lines = [`**Figure: ${spec.title}.**${spec.caption ? ` ${spec.caption}` : ''}`, ''];

    lines.push('Nodes:', '');
    for (const node of spec.nodes) {
        const extra = node.annotation ? `, ${node.annotation}` : '';
        lines.push(`- ${name.get(node.id)}: ${NODE_KIND_NAMES[node.kind]}${extra}`);
    }

    lines.push('', 'Edges:', '');
    for (const edge of spec.edges) {
        const kind = edge.kind ?? 'parent';
        const what =
            kind == 'binding' ? `binding ${ROLE_NAMES[edge.role!]}` : EDGE_KIND_NAMES[kind];
        const label = edge.label ? ` (“${edge.label}”)` : '';
        lines.push(`- ${name.get(edge.from)} → ${name.get(edge.to)}: ${what}${label}`);
    }
    return lines.join('\n');
}
