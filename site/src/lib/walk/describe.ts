import { hasTex } from '$lib/tex';
import { codeMatches, edgeId, matchLines, showMatch, type WalkthroughSpec } from './spec';

// A walkthrough as Markdown text, for the `.md` exports: each step's note, the code it points at,
// and what appears, changes kind, goes, or carries a pass, by the figure's own names.
export function describeWalkthrough(spec: WalkthroughSpec): string {
    const quote = (text: string) => (hasTex(text) || text.includes('`') ? text : `\`${text}\``);
    const nodes = new Map(spec.nodes.map((n) => [n.id, n]));
    const regions = new Map((spec.regions ?? []).map((r) => [r.id, r]));
    const edges = new Map(spec.edges.map((e) => [edgeId(e), e]));
    const name = (id: string): string => {
        const node = nodes.get(id);
        if (node) return quote(node.label ?? node.id);
        const region = regions.get(id);
        if (region) return quote(region.label ?? region.id);
        const edge = edges.get(id)!;
        return `${name(edge.from)} → ${name(edge.to)}`;
    };
    const kindName = (id: string, kind: string) =>
        (nodes.has(id) ? spec.nodeKinds[kind]?.name : spec.edgeKinds?.[kind]?.name) ?? kind;
    const list = (ids: Iterable<string>) => [...ids].map(name).join(', ');

    const lines = [`**Walkthrough: ${spec.title}.**${spec.caption ? ` ${spec.caption}` : ''}`, ''];
    spec.steps.forEach((step, i) => {
        const parts: string[] = [];
        const code = codeMatches(step);
        if (code.length) parts.push(`at ${code.map((m) => `\`${showMatch(m)}\``).join(', ')}`);
        if (step.show == 'all') parts.push('everything appears');
        else if (step.show?.some((id) => !edges.has(id))) {
            const shown = step.show.filter((id) => !edges.has(id));
            parts.push(
                `appears: ${shown
                    .map((id) => {
                        const node = nodes.get(id);
                        const kind = step.kind?.[id] ?? node?.kind;
                        return node && kind ? `${name(id)} (${kindName(id, kind)})` : name(id);
                    })
                    .join(', ')}`,
            );
        }
        const changes = Object.entries(step.kind ?? {}).filter(
            ([id]) => step.show == 'all' || !step.show?.includes(id),
        );
        if (changes.length) {
            parts.push(
                `becomes: ${changes.map(([id, kind]) => `${name(id)} → ${kindName(id, kind)}`).join(', ')}`,
            );
        }
        if (step.hide?.length) parts.push(`goes: ${list(step.hide)}`);
        if (step.mute?.length) parts.push(`set aside: ${list(step.mute)}`);
        if (step.flow?.length) parts.push(`flows along: ${list(step.flow)}`);
        if (step.flowBack?.length) parts.push(`flows back along: ${list(step.flowBack)}`);
        const detail = parts.length ? ` (${parts.join('; ')})` : '';
        lines.push(`${i + 1}. ${step.note}${detail}`);
    });
    return lines.join('\n');
}

/** The steps' code matches that find no line of `code`, as messages; empty when all do. */
export function staleMatches(spec: WalkthroughSpec, code: string): string[] {
    const lines = code.split('\n');
    return spec.steps.flatMap((step, i) =>
        codeMatches(step)
            .filter((match) => !matchLines(lines, match).length)
            .map(
                (match) =>
                    `step ${i + 1}: \`${showMatch(match)}\` matches no line of the quoted code`,
            ),
    );
}
