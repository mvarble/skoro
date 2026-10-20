import { cms } from '$cms';

import { CRATES } from '$lib/crates';
import { crateNames } from '$lib/server/reading';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = () => ({
    landing: cms.landing(),
    crates: CRATES.length > 0,
    reading: crateNames().length > 0,
});
