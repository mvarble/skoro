import { cms } from '$cms';

import { CRATES } from '$lib/crates';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = () => ({ landing: cms.landing(), crates: CRATES.length > 0 });
