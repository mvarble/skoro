// The entry point `bin/reading-order` runs: `bin/reading-order <crate> [--last a,b] [--json]`, where
// `<crate>` is a crate's directory, such as `crates/skoro-core`.
import path from 'node:path';

import { DEFAULT_LAST, readCrate, readingOrder } from './lib.ts';

// A reader that stops early, as `head` does, is not an error.
process.stdout.on('error', (error: NodeJS.ErrnoException) => {
    if (error.code == 'EPIPE') process.exit(0);
    throw error;
});

const args = process.argv.slice(2);
const flag = (name: string) => {
    const i = args.indexOf(name);
    return i < 0 ? undefined : (args.splice(i, 2)[1] ?? '');
};
const json = args.includes('--json');
if (json) args.splice(args.indexOf('--json'), 1);
const last = flag('--last')?.split(',').filter(Boolean) ?? DEFAULT_LAST;
const [crate] = args;
if (!crate || args.length != 1) {
    process.stderr.write('usage: bin/reading-order <crate directory> [--last a,b] [--json]\n');
    process.exit(2);
}

const order = readingOrder(readCrate(path.join(crate, 'src')), { last });
const name = path.basename(path.resolve(crate));
const count = (n: number, noun: string) => `${n} ${noun}${n == 1 ? '' : 's'}`;
if (json) {
    process.stdout.write(JSON.stringify(order, null, 2) + '\n');
} else if (order.modules.length < 2) {
    // A crate of one module, or of re-exports alone, has no order to read.
    const one = order.modules.length == 1 ? `one module, \`${order.modules[0]}\`` : 'no modules';
    process.stdout.write(`${name}: ${one} of its own; nothing to order.\n`);
} else {
    const at = new Map(order.modules.map((m, i) => [m, i]));
    const width = Math.max(...order.modules.map((m) => m.length));
    const digits = String(order.modules.length).length;
    process.stdout.write(
        `${name}: ${order.modules.length} modules, ` +
            `${count(order.forward.length, 'use')} of a later module (${count(order.cost, 'item')})` +
            `${order.exact ? '' : ', some cycle ordered greedily'}.\n\n`,
    );
    for (const [i, m] of order.modules.entries()) {
        const later = order.forward
            .filter((u) => u.from == m)
            .sort((a, b) => at.get(a.to)! - at.get(b.to)!)
            .map((u) => `${u.items.join(', ')} (${u.to})`);
        const line = `${String(i + 1).padStart(digits)}  ${m.padEnd(width)}`;
        process.stdout.write(
            `${later.length ? `${line}  uses later: ${later.join('; ')}` : line.trimEnd()}\n`,
        );
    }
}
