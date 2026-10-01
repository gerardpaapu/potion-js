#!/usr/bin/env -S node --experimental-strip-types --no-warnings=ExperimentalWarning

import { main } from './src/example.ts';

main().catch(e => {
    console.error(e);
    process.exitCode = 1;
})