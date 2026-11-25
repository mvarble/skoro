import { docsSite } from '@mvarble/mesearch-workflow/site';
import { defineConfig } from 'vitest/config';

// The documentation site, as `@mvarble/mesearch-workflow` builds every project's: the content
// database over `content/` (or `$SITE_CONTENT`, as `SITE_CONTENT=test/content pnpm build` builds
// the fixture), SvelteKit with mdsvex, and `$site`, which hands the shared pages skoro's own
// (`src/lib/site.ts`, `src/lib/server/site.ts`, the logo and `katex.config.ts`).
export default defineConfig({
    plugins: await docsSite(),
    test: {
        include: ['src/**/*.test.ts'],
    },
});
