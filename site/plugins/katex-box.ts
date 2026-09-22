import type { Plugin } from 'unified';
import type { Root } from 'hast';

type RehypeContent = Root['children'][number];

// Copied from the blog's `cms/src/plugins/mdsvex.ts`. It wraps each rendered display equation in a
// horizontally scrolling container and lifts a `\tag{..}` out beside it, so that a long equation
// scrolls without taking its label with it.
export const rehypeKatexBox: Plugin<[]> = () => {
    interface VisitState {
        parent: { children: RehypeContent[] };
        child: RehypeContent;
        index: number;
    }

    function visitor({ parent, child, index }: VisitState) {
        // find any katex display block
        if (
            child.type != 'element' ||
            child.tagName != 'div' ||
            !child.properties ||
            !('className' in child.properties) ||
            !Array.isArray(child.properties.className) ||
            !child.properties.className.includes('math') ||
            !child.properties.className.includes('math-display')
        ) {
            return;
        }

        // See if it has a tag.
        // If so, extract it and stick it higher up the DOM.
        // (super dirty, but what the hell)
        let tag: string | undefined = undefined;
        if (
            child.children.length == 1 &&
            child.children[0].type == 'text' &&
            child.children[0].value.startsWith('{@html')
        ) {
            const startIndex = child.children[0].value.indexOf('<span class=\\"katex-tag\\"');
            if (startIndex > 0) {
                // We now know we have a `<span class="katex-tag"` somewhere in the string; we need to
                // find the closing `</span>`.
                //
                // We do this by incrementing an index as we search for `</span>`, but keep track
                // of new opening `<span` blocks so that we don't short-circuit on a child.
                //
                // *Also* the first child of the tag block is a span with a 'katex-strut' class which
                // messes up the sizing once we move it up the DOM. That said, we drop the first
                // child by keeping track of *another* index.
                const rest = child.children[0].value.slice(startIndex);
                let length = 0;
                let depth = 0;
                let strutStartIndex = 0;
                let strutStopIndex = 0;
                while (length < rest.length) {
                    // if we are at the start of a `<span`, we increment the depth and track
                    // the indices if it is the first child within depth 1.
                    if (rest.slice(length, length + 5) === '<span') {
                        if (depth == 1 && strutStartIndex == 0) strutStartIndex = length;
                        depth++;
                        length += 5;
                        // Skip to end of opening tag
                        while (length < rest.length && rest[length] !== '>') length++;
                        length++; // Skip the '>'
                    }
                    // if we are at the start of a `</span>`, we decrement the depth and break
                    // if we have no depth.
                    else if (rest.slice(length, length + 7) === '</span>') {
                        depth--;
                        length += 7;
                        if (strutStartIndex != 0 && strutStopIndex == 0) strutStopIndex = length;
                        if (depth === 0) {
                            break;
                        }
                    } else {
                        length++;
                    }
                }

                const beforeText = child.children[0].value.slice(0, startIndex);
                tag = rest.slice(0, strutStartIndex) + rest.slice(strutStopIndex, length);
                const afterText = child.children[0].value.slice(startIndex + length);
                child.children[0].value = beforeText + afterText;
            }
        }

        const mathDiv: RehypeContent = {
            type: 'element',
            tagName: 'div',
            properties: {
                className: ['math-scroll-container'],
            },
            children: [
                {
                    type: 'element',
                    tagName: 'div',
                    properties: {
                        className: ['math-container'],
                    },
                    children: [child],
                },
            ],
        };

        if (tag) {
            parent.children[index] = {
                type: 'element',
                tagName: 'div',
                properties: {
                    className: ['math-tag-container'],
                },
                children: [
                    mathDiv,
                    {
                        type: 'element',
                        tagName: 'div',
                        properties: {
                            className: ['math-tag'],
                        },
                        children: [
                            {
                                type: 'text',
                                value: `{@html "${tag}"}`,
                            },
                        ],
                    },
                ],
            };
        } else {
            parent.children[index] = mathDiv;
        }
    }

    return (tree) => {
        if (!('children' in tree) || !Array.isArray(tree.children)) {
            return;
        }

        const n = tree.children.length - 1;
        const nodes: VisitState[] = tree.children.toReversed().map(
            (child, i) =>
                ({
                    parent: tree,
                    child,
                    index: n - i,
                }) as VisitState,
        );

        while (nodes.length > 0) {
            const node = nodes.pop()!;
            visitor(node);
            if ('children' in node.child && Array.isArray(node.child.children)) {
                const n = node.child.children.length - 1;
                nodes.push(
                    ...node.child.children.toReversed().map(
                        (child: RehypeContent, i: number) =>
                            ({
                                parent: node.child,
                                child,
                                index: n - i,
                            }) as VisitState,
                    ),
                );
            }
        }
    };
};
