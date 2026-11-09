import { cms } from '$cms';

import type { SkoroDocument } from '$lib/documents';

// A document's title where it is listed away from its entry: an archive document is named by its
// entry and its own title ("The core: Mathematics"), as the map names it; any other by its title.
export function qualifiedTitle(doc: SkoroDocument): string {
    return doc.kind == 'entry-document' ? `${cms.get(doc.entry)!.title}: ${doc.title}` : doc.title;
}
