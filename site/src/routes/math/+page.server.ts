import { cms } from '$cms';

import { linkOf } from '$lib/documents';
import { documentMap } from '$lib/server/map';
import type { PageServerLoad } from './$types';

// The math notes, and the map they make with the archive's documents that build on them.
export const load: PageServerLoad = () => ({
    notes: cms.math.list().map(linkOf),
    map: documentMap(),
});
