// The reading order, on a throwaway crate written to exercise each way one module uses another.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { readCrate, readingOrder, type Dependencies } from './lib.ts';

function crate(files: Record<string, string>): string {
    const src = fs.mkdtempSync(path.join(os.tmpdir(), 'reading-order-'));
    for (const [rel, text] of Object.entries(files)) {
        fs.mkdirSync(path.dirname(path.join(src, rel)), { recursive: true });
        fs.writeFileSync(path.join(src, rel), text);
    }
    return src;
}

const items = (deps: Dependencies, from: string, to: string) =>
    deps.uses.find((u) => u.from == from && u.to == to)?.items;

const sample = () =>
    readCrate(
        crate({
            'lib.rs': `
                mod alpha;
                mod beta;
                mod gamma;
                pub mod shapes;
                pub mod testing;
                pub use alpha::Alpha;
                pub use beta::{Beta, helper};
            `,
            'alpha.rs': `
                use crate::Beta;
                pub struct Alpha;
                pub fn take(_: Beta) {}
                // crate::gamma::Gamma, in a comment
                const S: &str = "crate::gamma::Gamma";
                const C: char = '"';
                #[cfg(test)]
                mod tests {
                    use crate::gamma::Gamma;
                }
            `,
            'beta.rs': `
                use crate::alpha;
                pub struct Beta;
                pub fn helper(a: alpha::Alpha) {
                    alpha::take(Beta);
                    a.widen();
                }
            `,
            'gamma.rs': `
                use crate::shapes::Round;
                pub struct Gamma;
                impl crate::alpha::Alpha {
                    pub fn widen(&self) {}
                }
                impl<'a> Round for Gamma {}
            `,
            'shapes/mod.rs': `
                mod round;
                pub use round::Round;
            `,
            'shapes/round.rs': `
                pub trait Round {}
                pub fn loose<Alpha: Copy>(a: Alpha) { a.widen(); }
            `,
            'testing.rs': `
                use crate::{Alpha, Beta};
                pub fn check(_: Alpha, _: Beta) {}
            `,
        }),
    );

test('reads the modules that define something, not the root or a module of re-exports', () => {
    assert.deepEqual(sample().modules, ['alpha', 'beta', 'gamma', 'shapes::round', 'testing']);
});

test('traces a re-exported name to the module that defines it', () => {
    const deps = sample();
    assert.deepEqual(items(deps, 'alpha', 'beta'), ['Beta']);
    assert.deepEqual(items(deps, 'gamma', 'shapes::round'), ['Round']);
    assert.deepEqual(items(deps, 'testing', 'alpha'), ['Alpha']);
});

test('counts the items used through a module imported whole', () => {
    assert.deepEqual(items(sample(), 'beta', 'alpha'), ['Alpha', 'take']);
});

test('finds a method another file adds to a type the caller names', () => {
    const deps = sample();
    assert.deepEqual(items(deps, 'beta', 'gamma'), ['Alpha.widen']);
    // `Alpha` is only a generic parameter there, so the call is not one to `gamma`'s method.
    assert.equal(items(deps, 'shapes::round', 'gamma'), undefined);
});

test('reads neither comments, strings nor test modules', () => {
    assert.equal(items(sample(), 'alpha', 'gamma'), undefined);
});

// x → y → z → x, where breaking z → x costs three items and either other use one.
const cycle: Dependencies = {
    modules: ['testing', 'x', 'y', 'z'],
    uses: [
        { from: 'x', to: 'y', items: ['Y'] },
        { from: 'y', to: 'z', items: ['Z'] },
        { from: 'z', to: 'x', items: ['A', 'B', 'C'] },
        { from: 'testing', to: 'x', items: ['A'] },
    ],
};

test('orders a cycle so that the fewest items are used ahead', () => {
    const order = readingOrder(cycle, { last: [] });
    assert.equal(order.cost, 1);
    assert.equal(order.exact, true);
    assert.equal(order.forward.length, 1);
    assert.ok(order.modules.indexOf('x') < order.modules.indexOf('z'));
});

test('reads test support last, and says when a cycle was too large to order exactly', () => {
    const order = readingOrder(cycle, { exactUpTo: 2 });
    assert.equal(order.modules.at(-1), 'testing');
    assert.equal(order.exact, false);
});
