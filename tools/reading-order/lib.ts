// The order in which to read a Rust crate's modules: what each module uses of the others, read
// from its source, and an order of the modules in which as little as possible is used before it is
// read. `main.ts` is the command line over it, and the site's `/reference/crates/<crate>/` pages
// draw it.
//
// A module uses another when it names one of its items: through a `use` (traced through re-exports
// to the module that defines the item), through an inline `crate::` or `super::` path, through a
// module imported whole (`use crate::scope;`, then `scope::build`), or by calling a method that
// another file's inherent `impl` adds to a type it names. Test modules (`#[cfg(test)] mod`) are not
// read, a `pub use` is a re-export and not a use, and a module that defines nothing of its own,
// such as `lib.rs` or a `mod.rs` of re-exports, is not one of the modules to read.
//
// This reads the text, not the types: a method called on a value whose type the calling module
// never names is missed, and a method sharing its name with one of an outside type's (`exp`,
// `sum`) is counted only where the type is named.
import fs from 'node:fs';
import path from 'node:path';

// ---------------------------------------------------------------------------
// Dependencies

/** One module's use of another: the items it names, each at most once. */
export interface Use {
    from: string;
    to: string;
    items: string[];
}

export interface Dependencies {
    /** The modules to read, as `scope` or `primitive::rule`, sorted by name. */
    modules: string[];
    uses: Use[];
}

type ItemPath = string[];

interface Statement {
    path: ItemPath;
    alias?: string;
    pub: boolean;
}

interface Module {
    id: string;
    /** The source without comments, string contents or test modules. */
    text: string;
    statements: Statement[];
    /** The source with its `use` statements removed, for the paths and calls written inline. */
    body: string;
}

/** Reads the crate whose sources are under `src` (the directory holding `lib.rs`). */
export function readCrate(src: string): Dependencies {
    const files = new Map<string, string>();
    for (const file of rustFiles(src)) {
        files.set(moduleOf(path.relative(src, file)), fs.readFileSync(file, 'utf8'));
    }
    return dependencies(files);
}

/** The dependencies among modules given as their source text, keyed by module (`''` for the root). */
export function dependencies(sources: Map<string, string>): Dependencies {
    const modules = new Map<string, Module>();
    for (const [id, source] of sources) {
        const text = withoutTestModules(stripped(source));
        const body = text.replace(USE, ' ');
        modules.set(id, { id, text, body, statements: statementsOf(text) });
    }

    const absolute = (from: string, p: ItemPath, depth = 0): ItemPath | undefined => {
        const [head, ...rest] = p;
        if (depth > 8) return undefined;
        if (head == 'crate' || head == '$crate') return rest;
        if (head == 'self') return [...segments(from), ...rest];
        if (head == 'super') {
            let at = segments(from);
            let tail = p;
            while (tail[0] == 'super') {
                at = at.slice(0, -1);
                tail = tail.slice(1);
            }
            return [...at, ...tail];
        }
        // A path relative to the module itself, as in `pub use discrete::Add;` in a `mod.rs`.
        const child = [...segments(from), head!];
        if (modules.has(child.join('::'))) return [...child, ...rest];
        // A name the module imports, as `scope` in `impl scope::Draft` after `use crate::scope;`.
        const imported = modules
            .get(from)
            ?.statements.find((s) => (s.alias ?? s.path.at(-1)) == head && s.path.length > 1);
        return imported ? absolute(from, [...imported.path, ...rest], depth + 1) : undefined;
    };

    // What each module re-exports, by the name it re-exports under.
    const reexports = new Map<string, Map<string, ItemPath>>();
    for (const module of modules.values()) {
        for (const s of module.statements) {
            const target = s.pub ? absolute(module.id, s.path) : undefined;
            if (!target?.length) continue;
            const names = reexports.get(module.id) ?? new Map<string, ItemPath>();
            names.set(s.alias ?? target.at(-1)!, target);
            reexports.set(module.id, names);
        }
    }

    // The module defining what a path names, and the name of the item in it (undefined for a
    // module named whole).
    const resolve = (p: ItemPath, depth = 0): { module: string; item?: string } | undefined => {
        if (depth > 16) return undefined;
        for (let k = p.length; k >= 0; k--) {
            const id = p.slice(0, k).join('::');
            if (!modules.has(id)) continue;
            const rest = p.slice(k);
            const via = rest.length ? reexports.get(id)?.get(rest[0]!) : undefined;
            if (via) return resolve([...via, ...rest.slice(1)], depth + 1);
            return { module: id, item: rest[0] };
        }
        return undefined;
    };

    const facade = (id: string) => !DEFINES.test(modules.get(id)!.body);
    const items = new Map<string, Set<string>>();
    // What each module names, as `module\nitem`: the types whose methods it can be calling.
    const known = new Map<string, Set<string>>();
    const add = (from: string, to: string, item: string) => {
        if (from == to || facade(from) || facade(to)) return;
        const key = `${from}\n${to}`;
        items.set(key, (items.get(key) ?? new Set()).add(item));
    };

    for (const module of modules.values()) {
        const named: ItemPath[] = [];
        for (const s of module.statements) {
            if (s.pub) continue;
            const target = absolute(module.id, s.path);
            if (!target) continue;
            const found = resolve(target);
            if (found && found.item === undefined && found.module != module.id) {
                // A module imported whole: count what is used through it.
                const alias = escape(s.alias ?? s.path.at(-1)!);
                for (const [, name] of module.body.matchAll(
                    new RegExp(`(?<![\\w:])${alias}::(\\w+)`, 'g'),
                )) {
                    named.push([...target, name!]);
                }
            } else {
                named.push(target);
            }
        }
        for (const [, kind, rest] of module.body.matchAll(INLINE)) {
            const p = [kind!.replace('$', ''), ...rest!.split('::').filter(Boolean)];
            const target = absolute(module.id, p);
            if (target) named.push(target);
        }
        const names = new Set<string>();
        for (const p of named) {
            const found = resolve(p);
            if (found?.item === undefined) continue;
            add(module.id, found.module, found.item);
            names.add(`${found.module}\n${found.item}`);
        }
        known.set(module.id, names);
    }

    // Methods that one file's inherent `impl` adds to a type another file defines.
    const definedIn = new Map<string, string[]>();
    for (const module of modules.values()) {
        for (const [, name] of module.text.matchAll(DEFINITION)) {
            definedIn.set(name!, [...(definedIn.get(name!) ?? []), module.id]);
        }
    }
    for (const module of modules.values()) {
        const methods: { type: string; owner: string; method: string }[] = [];
        for (const match of module.text.matchAll(IMPL)) {
            const typePath = match[2]!.split('::');
            const type = typePath.at(-1)!;
            const owner =
                typePath.length > 1
                    ? resolve(absolute(module.id, typePath) ?? [])?.module
                    : definedIn.get(type)?.length == 1
                      ? definedIn.get(type)![0]
                      : undefined;
            if (owner === undefined || owner == module.id) continue;
            const block = braced(module.text, match.index! + match[0].length - 1);
            for (const [, method] of block.matchAll(/\bfn\s+(\w+)/g)) {
                methods.push({ type, owner, method: method! });
            }
        }
        // A call counts where the caller names the type (it imports or defines it) and calls a
        // method of that name. A name the file gives to methods of several types is not told apart.
        for (const { type, owner, method } of methods) {
            const shared = methods.filter((m) => m.method == method).length > 1;
            const call = new RegExp(`[.:]${escape(method)}\\s*(::<[^>]*>)?\\s*\\(`);
            for (const caller of modules.values()) {
                if (caller.id == module.id) continue;
                const names = caller.id == owner || known.get(caller.id)!.has(`${owner}\n${type}`);
                if (names && call.test(caller.body)) {
                    add(caller.id, module.id, shared ? `.${method}` : `${type}.${method}`);
                }
            }
        }
    }

    // The root module, should it define anything, is read as `crate`.
    const name = (id: string) => id || 'crate';
    const uses = [...items]
        .map(([key, set]) => {
            const [from, to] = key.split('\n') as [string, string];
            return { from: name(from), to: name(to), items: [...set].sort() };
        })
        .sort((a, b) => a.from.localeCompare(b.from) || a.to.localeCompare(b.to));
    const read = [...modules.keys()]
        .filter((id) => !facade(id))
        .map(name)
        .sort();
    return { modules: read, uses };
}

// ---------------------------------------------------------------------------
// The order

export interface ReadingOrder {
    /** The modules, in the order to read them. */
    modules: string[];
    uses: Use[];
    /** The uses of a module that comes later in the order. */
    forward: Use[];
    /** The number of items those uses name: what the order minimises. */
    cost: number;
    /** Whether every cycle was small enough to be ordered exactly. */
    exact: boolean;
}

// Test support, which uses whatever it tests, is read last unless told otherwise.
export const DEFAULT_LAST = ['testing'];

export interface OrderOptions {
    /** Modules read after all others, in this order, such as test support. */
    last?: string[];
    /** The largest cycle ordered exactly; a larger one is ordered greedily. */
    exactUpTo?: number;
}

// Orders the modules so that the items used from later modules are as few as can be, then, among
// the orders that achieve it, reads each module as soon as possible after the last module it uses,
// and leaves for the end the modules no other module uses.
//
// Every cycle of the dependencies (each strongly connected component) is ordered on its own: an
// order of the components that follows the dependencies costs nothing, so the minimum is the sum of
// each component's minimum, which a dynamic programme over the subsets of the component finds
// exactly.
export function readingOrder(deps: Dependencies, options: OrderOptions = {}): ReadingOrder {
    const last = (options.last ?? []).filter((m) => deps.modules.includes(m));
    const exactUpTo = options.exactUpTo ?? 20;
    const modules = deps.modules.filter((m) => !last.includes(m));
    const weight = new Map(deps.uses.map((u) => [`${u.from}\n${u.to}`, u.items.length]));
    const w = (a: string, b: string) => weight.get(`${a}\n${b}`) ?? 0;
    const users = new Map<string, string[]>(modules.map((m) => [m, []]));
    const used = new Map<string, string[]>(modules.map((m) => [m, []]));
    for (const u of deps.uses) {
        if (!users.has(u.from) || !users.has(u.to)) continue;
        used.get(u.from)!.push(u.to);
        users.get(u.to)!.push(u.from);
    }

    let exact = true;
    const first: string[] = [];
    for (const component of components(modules, used)) {
        if (component.length == 1) {
            first.push(component[0]!);
        } else if (component.length <= exactUpTo) {
            first.push(...cheapest(component, w));
        } else {
            exact = false;
            first.push(...greedy(component, w));
        }
    }

    // Among orders of the same cost: as soon after its last prerequisite as possible, the modules
    // nothing uses last, and otherwise the order found.
    const position = new Map(first.map((m, i) => [m, i]));
    const before = (m: string) => used.get(m)!.filter((d) => position.get(d)! < position.get(m)!);
    const leaf = (m: string) => users.get(m)!.length == 0;
    const order: string[] = [];
    const placed = new Map<string, number>();
    while (order.length < first.length) {
        const ready = first.filter((m) => !placed.has(m) && before(m).every((d) => placed.has(d)));
        const key = (m: string): [number, number, number] => [
            leaf(m) ? 0 : 1,
            Math.max(-1, ...before(m).map((d) => placed.get(d)!)),
            -position.get(m)!,
        ];
        const next = ready.reduce((a, b) => (compare(key(b), key(a)) > 0 ? b : a));
        placed.set(next, order.length);
        order.push(next);
    }
    order.push(...last);

    const at = new Map(order.map((m, i) => [m, i]));
    const forward = deps.uses.filter((u) => at.get(u.to)! > at.get(u.from)!);
    const cost = forward.reduce((sum, u) => sum + u.items.length, 0);
    return { modules: order, uses: deps.uses, forward, cost, exact };
}

// The strongly connected components, each listed once, dependencies before what uses them.
function components(modules: string[], used: Map<string, string[]>): string[][] {
    const index = new Map<string, number>();
    const low = new Map<string, number>();
    const stack: string[] = [];
    const onStack = new Set<string>();
    const out: string[][] = [];
    const visit = (v: string) => {
        index.set(v, index.size);
        low.set(v, index.get(v)!);
        stack.push(v);
        onStack.add(v);
        for (const u of used.get(v)!) {
            if (!index.has(u)) {
                visit(u);
                low.set(v, Math.min(low.get(v)!, low.get(u)!));
            } else if (onStack.has(u)) {
                low.set(v, Math.min(low.get(v)!, index.get(u)!));
            }
        }
        if (low.get(v) == index.get(v)) {
            const component: string[] = [];
            let u: string;
            do {
                u = stack.pop()!;
                onStack.delete(u);
                component.push(u);
            } while (u != v);
            out.push(component.sort());
        }
    };
    for (const m of modules) if (!index.has(m)) visit(m);
    return out;
}

// The order of one component that minimises the items used from later modules: the cheapest
// way to place the modules of each subset first, built up one module at a time.
function cheapest(component: string[], w: (a: string, b: string) => number): string[] {
    const n = component.length;
    const size = 1 << n;
    // Each module's uses within the component, as (bit, weight) pairs.
    const out = component.map((a) =>
        component.flatMap((b, j) => (w(a, b) ? [[1 << j, w(a, b)] as const] : [])),
    );
    const best = new Float64Array(size).fill(Infinity);
    const choice = new Int8Array(size).fill(-1);
    best[0] = 0;
    for (let set = 0; set < size; set++) {
        const so_far = best[set]!;
        for (let v = 0; v < n; v++) {
            const bit = 1 << v;
            if (set & bit) continue;
            // Placing `v` next costs what it uses that is not yet placed.
            let cost = so_far;
            for (const [b, weight] of out[v]!) if (!(set & b)) cost += weight;
            if (cost < best[set | bit]!) {
                best[set | bit] = cost;
                choice[set | bit] = v;
            }
        }
    }
    const order: string[] = [];
    for (let set = size - 1; set;) {
        const v = choice[set]!;
        order.push(component[v]!);
        set &= ~(1 << v);
    }
    return order.reverse();
}

// A component too large to order exactly: repeatedly place the module whose uses of the modules
// not yet placed weigh least.
function greedy(component: string[], w: (a: string, b: string) => number): string[] {
    const left = new Set(component);
    const order: string[] = [];
    while (left.size) {
        const cost = (m: string) => [...left].reduce((s, u) => s + w(m, u), 0);
        const next = [...left].reduce((a, b) => (cost(b) < cost(a) ? b : a));
        left.delete(next);
        order.push(next);
    }
    return order;
}

function compare(a: number[], b: number[]): number {
    for (let i = 0; i < a.length; i++) if (a[i] != b[i]) return a[i]! - b[i]!;
    return 0;
}

// ---------------------------------------------------------------------------
// Reading Rust source

const USE = /\b(pub(?:\s*\([^)]*\))?\s+)?use\s+([^;]+);/g;
const INLINE = /(?<![\w:])(\$?crate|super)((?:::\w+)+)/g;
const CHAR = /^'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^\\'])'/;
const DEFINES = /\b(fn|struct|enum|trait|impl|const|static|type|union|macro_rules!)\b/;
const DEFINITION = /\b(?:struct|enum|trait|type|union)\s+([A-Z]\w*)/g;
// An inherent `impl`: a type, possibly generic, then the block (a trait impl has `for` instead).
const GENERICS = String.raw`<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>`;
const IMPL = new RegExp(
    String.raw`\bimpl\s*(${GENERICS})?\s+((?:\w+::)*[A-Z]\w*)\s*(?:${GENERICS})?\s*(?:where[^{;]*)?\{`,
    'g',
);

function* rustFiles(dir: string): Generator<string> {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const file = path.join(dir, entry.name);
        if (entry.isDirectory()) yield* rustFiles(file);
        else if (entry.name.endsWith('.rs')) yield file;
    }
}

// `lib.rs` is the root (`''`), `primitive/mod.rs` is `primitive`, `primitive/rule.rs` is
// `primitive::rule`.
export function moduleOf(relative: string): string {
    const parts = relative.split(path.sep).join('/').replace(/\.rs$/, '').split('/');
    if (parts.at(-1) == 'mod' || (parts.length == 1 && parts[0] == 'lib')) parts.pop();
    return parts.join('::');
}

const segments = (id: string) => (id ? id.split('::') : []);
const escape = (s: string) => s.replace(/[$.*+?^(){}[\]|\\]/g, '\\$&');

// The source with comments removed and string literals emptied, so that neither is read as code.
function stripped(source: string): string {
    let out = '';
    let i = 0;
    while (i < source.length) {
        const c = source[i]!;
        const next = source[i + 1];
        if (c == '/' && next == '/') {
            while (i < source.length && source[i] != '\n') i++;
        } else if (c == '/' && next == '*') {
            let depth = 0;
            do {
                if (source.startsWith('/*', i)) (depth++, (i += 2));
                else if (source.startsWith('*/', i)) (depth--, (i += 2));
                else i++;
            } while (depth > 0 && i < source.length);
            out += ' ';
        } else if (
            c == 'r' &&
            /^r#*"/.test(source.slice(i, i + 8)) &&
            !/\w/.test(source[i - 1] ?? '')
        ) {
            const hashes = /^r(#*)"/.exec(source.slice(i))![1]!;
            const end = source.indexOf(`"${hashes}`, i + hashes.length + 2);
            i = end < 0 ? source.length : end + hashes.length + 1;
            out += '""';
        } else if (c == '"') {
            i++;
            while (i < source.length && source[i] != '"') i += source[i] == '\\' ? 2 : 1;
            i++;
            out += '""';
        } else if (c == "'" && CHAR.test(source.slice(i, i + 12))) {
            // A character literal (a lifetime has no closing quote).
            i += CHAR.exec(source.slice(i, i + 12))![0].length;
            out += "' '";
        } else {
            out += c;
            i++;
        }
    }
    return out;
}

// Removes each `#[cfg(test)] mod name { … }`.
function withoutTestModules(text: string): string {
    const pattern =
        /#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{/g;
    let out = '';
    let from = 0;
    for (const match of text.matchAll(pattern)) {
        if (match.index! < from) continue;
        out += text.slice(from, match.index);
        from =
            match.index! +
            match[0].length -
            1 +
            braced(text, match.index! + match[0].length - 1).length +
            2;
    }
    return out + text.slice(from);
}

// The text between the brace at `open` and the brace that closes it.
function braced(text: string, open: number): string {
    let depth = 0;
    for (let i = open; i < text.length; i++) {
        if (text[i] == '{') depth++;
        else if (text[i] == '}' && --depth == 0) return text.slice(open + 1, i);
    }
    return text.slice(open + 1);
}

function statementsOf(text: string): Statement[] {
    const out: Statement[] = [];
    for (const [, pub, tree] of text.matchAll(USE)) {
        for (const { path: p, alias } of expand(tree!.replace(/\s+/g, ' ').trim(), [])) {
            if (p.at(-1) == '*') continue;
            out.push({ path: p, ...(alias ? { alias } : {}), pub: Boolean(pub) });
        }
    }
    return out;
}

// A use tree, `a::{b, c::{self, D as E}}`, as the paths it imports.
function expand(tree: string, prefix: ItemPath): { path: ItemPath; alias?: string }[] {
    const open = tree.indexOf('{');
    if (open >= 0 && tree.endsWith('}')) {
        const head = tree
            .slice(0, open)
            .split('::')
            .map((s) => s.trim())
            .filter(Boolean);
        return splitTop(tree.slice(open + 1, -1)).flatMap((sub) =>
            expand(sub, [...prefix, ...head]),
        );
    }
    const [, name, alias] = /^(.*?)(?:\s+as\s+(\w+))?$/.exec(tree.trim())!;
    const parts = name!
        .split('::')
        .map((s) => s.trim())
        .filter(Boolean);
    if (parts.length == 1 && parts[0] == 'self')
        return [{ path: prefix, ...(alias ? { alias } : {}) }];
    return [{ path: [...prefix, ...parts], ...(alias ? { alias } : {}) }];
}

function splitTop(list: string): string[] {
    const out: string[] = [];
    let depth = 0;
    let current = '';
    for (const c of list) {
        if (c == '{') depth++;
        if (c == '}') depth--;
        if (c == ',' && depth == 0) {
            out.push(current);
            current = '';
        } else {
            current += c;
        }
    }
    out.push(current);
    return out.map((s) => s.trim()).filter(Boolean);
}
