import * as FS from 'node:fs/promises';

const hasOwn = Object.prototype.hasOwnProperty;

export interface TokenizerJson {
    model: {
        // the vocab is a mapping of strings to their token ids
        vocab: Record<string, number>
    }
}

export async function loadTokenizer(path: string) {
    const raw = await FS.readFile(path, 'utf-8');
    return JSON.parse(raw);
}

export function tokenize(source: string, config: TokenizerJson) {
    const tokens = source.split(/\s+/);
    const vocab = config.model.vocab as Record<string, number>;

    const ids = [];
    for (const token of tokens) {
        if (token === '') {
            continue;
        }
        let i = token.length;
        let start = 0;

        // This is probably not a good way to do this, but I want to take
        // the longest prefix I can from the current word 
        // (including the "prefix" that is the whole word)
        while (i > start) {
            let chunk = token.slice(start, i);
            if (start > 0) {
                chunk = `##${chunk}`;
            }

            if (hasOwn.call(vocab, chunk)) {
                ids.push(vocab[chunk]);
                start = i;
                i = token.length;
                continue;
            }
            i--;
        }

        if (i === 0) {
            let codepoints = Array.from(token, (cha) => cha.codePointAt(0));
            throw new Error(`Couldn't tokenize, no entry for: ${token} (${JSON.stringify(codepoints)})`);
        }
    }

    return ids;
}
