import fs from 'node:fs';
import path from 'node:path';

import type { ReadingOrderSpec } from '$lib/reading/spec';
import { REPO_ROOT } from '../../../plugins/file-ref';
import { DEFAULT_LAST, readCrate, readingOrder } from '../../../../tools/reading-order/lib.ts';

// The reading orders of the workspace's crates, computed from their sources when the site is
// built (and on every request under `pnpm dev`), so the pages follow the code as it is.

export interface CrateReading {
    crate: string;
    modules: number;
    /** The uses of a later module, and the items they name. */
    ahead: number;
    items: number;
    /** Undefined for a crate with fewer than two modules of its own: there is no order to read. */
    spec?: ReadingOrderSpec;
}

/** The crates under `crates/`, each a directory with a `src/lib.rs`. */
export function crateNames(): string[] {
    const dir = path.join(REPO_ROOT, 'crates');
    if (!fs.existsSync(dir)) return [];
    return fs
        .readdirSync(dir)
        .filter((name) => fs.existsSync(path.join(dir, name, 'src', 'lib.rs')))
        .sort();
}

export function readingOf(crate: string): CrateReading | undefined {
    if (!crateNames().includes(crate)) return undefined;
    const order = readingOrder(readCrate(path.join(REPO_ROOT, 'crates', crate, 'src')), {
        last: DEFAULT_LAST,
    });
    const reading = {
        crate,
        modules: order.modules.length,
        ahead: order.forward.length,
        items: order.cost,
    };
    if (order.modules.length < 2) return reading;
    const count = (n: number, noun: string) => `${n} ${noun}${n == 1 ? '' : 's'}`;
    return {
        ...reading,
        spec: {
            title: `The order in which to read ${crate}`,
            caption:
                `${count(order.forward.length, 'use')} of a later module, naming ` +
                `${count(order.cost, 'item')}: ` +
                (order.exact
                    ? 'as few as any order allows.'
                    : 'a cycle too large to order exactly was ordered greedily, so there may be fewer.'),
            modules: order.modules.map((id) => ({ id, apart: DEFAULT_LAST.includes(id) })),
            uses: order.uses,
        },
    };
}
