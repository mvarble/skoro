# Fixture content

A small site with a document of every kind: pages in two tracks, math notes that build on each
other, a finished entry with its documents (one with a numbered equation, a statement and a
citation) and an entry in progress that revises it. CI builds it with

```sh
SITE_CONTENT=test/content pnpm build
```

so that every route renders something, which the real content may not do yet.
