import { crateNames, readingOf } from '$lib/server/reading';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = () => ({
    readings: crateNames().map((crate) => {
        const { modules, ahead, items } = readingOf(crate)!;
        return { crate, modules, ahead, items };
    }),
});
