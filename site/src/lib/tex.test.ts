import { describe, expect, it } from 'vitest';

import { hasTex, renderMixed, renderTex } from './tex';

describe('renderMixed', () => {
    it('escapes plain text', () => {
        expect(renderMixed('a < b & c')).toBe('a &lt; b &amp; c');
    });

    it('renders each $…$ span with KaTeX and keeps the text around it', () => {
        const html = renderMixed('score $s$ of $\\theta$');
        expect(html.startsWith('score <span class="katex">')).toBe(true);
        expect(html).toContain(' of <span class="katex">');
        expect(html.match(/class="katex"/g)).toHaveLength(2);
    });

    it('treats \\$ as a literal dollar sign', () => {
        expect(renderMixed('costs \\$5')).toBe('costs $5');
        expect(hasTex('costs \\$5')).toBe(false);
    });

    it('leaves an unmatched $ as text', () => {
        expect(renderMixed('a $ b')).toBe('a $ b');
    });

    it('applies the site-wide macros, and per-call macros over them', () => {
        expect(renderTex('\\R')).toContain('mathbb');
        expect(renderTex('\\foo', { macros: { '\\foo': 'y' } })).toContain('>y<');
    });

    it('does not let a \\gdef leak into the next call', () => {
        renderTex('\\gdef\\leak{z}');
        // KaTeX draws an undefined control sequence in its error colour.
        expect(renderTex('\\leak')).toContain('#cc0000');
    });
});
