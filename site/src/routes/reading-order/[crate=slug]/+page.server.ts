import fs from 'node:fs';
import path from 'node:path';

import { error } from '@sveltejs/kit';

import { dev } from '$app/environment';
import type { ReadingOrderSpec } from '$lib/reading/spec';
import { REPO_ROOT } from '../../../../plugins/file-ref';
import { DEFAULT_LAST, readCrate, readingOrder } from '../../../../../tools/reading-order/lib.ts';
import type { PageServerLoad } from './$types';

// The order in which to read a crate's modules, read from its sources on every request, for
// reviewing a build before it is published. Nothing links here, so the static build never
// renders it; it is the development server's alone.
export const load: PageServerLoad = ({ params }) => {
    if (!dev) error(404, 'Reading orders are drawn by the development server only.');
    const src = path.join(REPO_ROOT, 'crates', params.crate, 'src');
    if (!fs.existsSync(path.join(src, 'lib.rs'))) error(404, `No crate at crates/${params.crate}.`);

    const order = readingOrder(readCrate(src), { last: DEFAULT_LAST });
    const count = (n: number, noun: string) => `${n} ${noun}${n == 1 ? '' : 's'}`;
    const spec: ReadingOrderSpec = {
        title: `The order in which to read ${params.crate}`,
        caption:
            `${count(order.forward.length, 'use')} of a later module, naming ` +
            `${count(order.cost, 'item')}: ` +
            (order.exact
                ? 'as few as any order allows.'
                : 'a cycle too large to order exactly was ordered greedily, so there may be fewer.'),
        modules: order.modules.map((id) => ({ id, apart: DEFAULT_LAST.includes(id) })),
        uses: order.uses,
    };
    return { crate: params.crate, spec };
};
