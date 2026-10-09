import { cms } from '$cms';

import { linkOf } from '$lib/documents';
import type { PageServerLoad } from './$types';

// Entries in progress first, then those done, then those set aside; newest first within each.
export const load: PageServerLoad = () => {
    const entries = cms.archive.list();
    const group = (status: string) => entries.filter((e) => e.status == status).map(linkOf);
    return { active: group('active'), done: group('done'), abandoned: group('abandoned') };
};
