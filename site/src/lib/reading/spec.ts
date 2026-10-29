// The input to `<ReadingOrder>`: the modules of a crate in the order to read them, and what each
// uses of the others.
//
// The figure lists the modules top to bottom in the order given and draws each use as an arc: on
// the left when it reaches back to a module already read, on the right, dashed, when it reaches
// ahead to one not yet read, with the items it uses named beside the module. The order is the
// figure's to declare; `tools/reading-order` computes one that keeps the uses ahead few.

export interface ReadingModuleSpec {
    /** The module, as `scope` or `primitive::rule`. */
    id: string;
    /** Set apart below the rest, as test support is. Only the last modules may be. */
    apart?: boolean;
}

/** One module's use of another. */
export interface ReadingUseSpec {
    from: string;
    to: string;
    /** What it names of the other module: `Space`, `narrow`, `Lowering.compile`. */
    items: string[];
}

export interface ReadingOrderSpec {
    /** The figure's accessible name. */
    title: string;
    caption?: string;
    /** The modules, in the order to read them. */
    modules: ReadingModuleSpec[];
    uses: ReadingUseSpec[];
}

/** Checks a reading order, failing with its title in the message, and returns it. */
export function defineReadingOrder(spec: ReadingOrderSpec): ReadingOrderSpec {
    const fail = (message: string) => {
        throw new Error(`Reading order "${spec.title}": ${message}`);
    };
    const ids = new Set<string>();
    for (const m of spec.modules) {
        if (ids.has(m.id)) fail(`the module \`${m.id}\` is listed twice.`);
        ids.add(m.id);
    }
    const firstApart = spec.modules.findIndex((m) => m.apart);
    if (firstApart >= 0 && spec.modules.slice(firstApart).some((m) => !m.apart)) {
        fail('only the last modules may be set apart.');
    }
    for (const u of spec.uses) {
        for (const end of [u.from, u.to]) {
            if (!ids.has(end)) fail(`a use names \`${end}\`, which is not one of its modules.`);
        }
        if (u.from == u.to) fail(`\`${u.from}\` uses itself.`);
    }
    return spec;
}

/** An item as the figure shows it: a function or method with `()`. */
export function itemText(item: string): string {
    const last = item.split(/::|\./).at(-1) ?? item;
    return /^[a-z_]/.test(last) ? `${item}()` : item;
}
