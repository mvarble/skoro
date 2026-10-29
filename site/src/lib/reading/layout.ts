import { defineReadingOrder, itemText, type ReadingOrderSpec, type ReadingUseSpec } from './spec';

// The geometry of a reading-order figure, in SVG coordinates (+Y down): one row per module, the
// uses of earlier modules as arcs bowing left from the rows' dots, the uses of later modules as
// dashed arcs bowing right from the end of the names, and a band behind each cycle. Text widths
// are estimated from the monospace face's advance, since nothing is measured.

export const ROW = 26;
const TOP = 20;
const APART_GAP = 16;
/** The advance of the module names (13 units of JetBrains Mono) and of the items (11.5 units). */
const NAME_ADVANCE = 7.8;
const ITEM_ADVANCE = 6.9;

export interface RowLayout {
    id: string;
    /** From 1. */
    number: number;
    y: number;
    /** The items of later modules this one uses, as shown beside it. */
    later: string;
}

export interface UseLayout extends ReadingUseSpec {
    /** Whether `to` comes after `from`. */
    later: boolean;
    /** A use of an earlier module that a longer chain of such uses already implies. */
    implied: boolean;
    d: string;
}

export interface BandLayout {
    y: number;
    height: number;
    /** The number of modules in the cycle. */
    size: number;
}

export interface ReadingOrderLayout {
    width: number;
    height: number;
    rows: RowLayout[];
    /** Longest first, so the short arcs are drawn over the long ones. */
    uses: UseLayout[];
    bands: BandLayout[];
    /** Where the dots are, where the names start, and where the dashed arcs start. */
    dotX: number;
    nameX: number;
    laterX: number;
    itemX: number;
}

export function layoutReadingOrder(input: ReadingOrderSpec): ReadingOrderLayout {
    const spec = defineReadingOrder(input);
    const at = new Map(spec.modules.map((m, i) => [m.id, i]));
    const y = (id: string) => {
        const i = at.get(id)!;
        return TOP + i * ROW + (spec.modules[i]!.apart ? APART_GAP : 0);
    };
    const later = (u: ReadingUseSpec) => at.get(u.to)! > at.get(u.from)!;

    // A use of an earlier module is implied when `to` is reached from `from` by other such uses.
    const earlier = spec.uses.filter((u) => !later(u));
    const reaches = (from: string, to: string, skip: ReadingUseSpec) => {
        const seen = new Set<string>();
        const stack = earlier.filter((u) => u.from == from && u != skip).map((u) => u.to);
        while (stack.length) {
            const next = stack.pop()!;
            if (next == to) return true;
            if (seen.has(next)) continue;
            seen.add(next);
            stack.push(...earlier.filter((u) => u.from == next).map((u) => u.to));
        }
        return false;
    };

    const leftBulge = (span: number) => Math.min(285, 10 + 0.42 * span);
    const rightBulge = (span: number) => 12 + 0.17 * span;
    const span = (u: ReadingUseSpec) => Math.abs(y(u.to) - y(u.from));
    const widest = (xs: number[]) => Math.max(0, ...xs);

    const dotX = 0.75 * widest(earlier.map((u) => leftBulge(span(u)))) + 12;
    const nameX = dotX + 44;
    const laterX = nameX + widest(spec.modules.map((m) => m.id.length)) * NAME_ADVANCE + 18;
    const ahead = spec.uses.filter(later);
    const itemX = laterX + 0.75 * widest(ahead.map((u) => rightBulge(span(u)))) + 16;

    const rows = spec.modules.map((m, i) => ({
        id: m.id,
        number: i + 1,
        y: y(m.id),
        later: ahead
            .filter((u) => u.from == m.id)
            .sort((a, b) => at.get(a.to)! - at.get(b.to)!)
            .map((u) => u.items.map(itemText).join(', '))
            .join(' · '),
    }));

    const uses = spec.uses
        .map((u) => {
            const [y0, y1] = [y(u.from), y(u.to)];
            const isLater = later(u);
            const [x, k] = isLater ? [laterX, rightBulge(span(u))] : [dotX, -leftBulge(span(u))];
            return {
                ...u,
                later: isLater,
                implied: !isLater && reaches(u.from, u.to, u),
                d: `M${x} ${y0} C${round(x + k)} ${y0} ${round(x + k)} ${y1} ${x} ${y1}`,
            };
        })
        .sort((a, b) => span(b) - span(a));

    const bands = cycles(
        spec.modules.map((m) => m.id),
        spec.uses,
    ).flatMap((cycle) =>
        runs(cycle.map((id) => at.get(id)!)).map(([first, last]) => ({
            y: y(spec.modules[first]!.id) - ROW / 2 + 2,
            height: y(spec.modules[last]!.id) - y(spec.modules[first]!.id) + ROW - 4,
            size: cycle.length,
        })),
    );

    const lastY = rows.at(-1)?.y ?? TOP;
    return {
        width: Math.ceil(itemX + widest(rows.map((r) => r.later.length)) * ITEM_ADVANCE + 8),
        height: lastY + ROW,
        rows,
        uses,
        bands,
        dotX,
        nameX,
        laterX,
        itemX,
    };
}

const round = (x: number) => Math.round(x * 10) / 10;

// The cycles of the uses: the strongly connected components with more than one module.
export function cycles(modules: string[], uses: ReadingUseSpec[]): string[][] {
    const next = new Map(modules.map((m) => [m, uses.filter((u) => u.from == m).map((u) => u.to)]));
    const index = new Map<string, number>();
    const low = new Map<string, number>();
    const stack: string[] = [];
    const out: string[][] = [];
    const visit = (v: string) => {
        index.set(v, index.size);
        low.set(v, index.get(v)!);
        stack.push(v);
        for (const u of next.get(v)!) {
            if (!index.has(u)) {
                visit(u);
                low.set(v, Math.min(low.get(v)!, low.get(u)!));
            } else if (stack.includes(u)) {
                low.set(v, Math.min(low.get(v)!, index.get(u)!));
            }
        }
        if (low.get(v) == index.get(v)) {
            const component = stack.splice(stack.indexOf(v));
            if (component.length > 1) out.push(component);
        }
    };
    for (const m of modules) if (!index.has(m)) visit(m);
    return out;
}

// Positions as maximal runs of consecutive numbers, [first, last].
function runs(positions: number[]): [number, number][] {
    const sorted = [...positions].sort((a, b) => a - b);
    const out: [number, number][] = [];
    for (const p of sorted) {
        const run = out.at(-1);
        if (run && run[1] == p - 1) run[1] = p;
        else out.push([p, p]);
    }
    return out;
}
