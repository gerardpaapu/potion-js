import * as Path from 'node:path/posix';
import { existsSync, } from 'node:fs';
import { loadTokenizer, tokenize } from './tokenizer.ts'
import { loadModel } from './loader.ts';
import { encode } from './encoder.ts';
import { quantizeTo1Bit } from './quantize.ts';

import * as FS from 'node:fs/promises';

export async function main() {
    const MODELS_PATH = process.env.MODELS_PATH ?? Path.join(process.env.HOME!, '.local/share/models');
    const potionPath = Path.join(MODELS_PATH, 'potion-mxbai-micro');
    const modelPath = Path.join(potionPath, 'model.safetensors');
    const tokenizerPath = Path.join(potionPath, 'tokenizer.json');

    if (!existsSync(modelPath)) {
        throw new Error(`Missing model file at: ${modelPath}`);
    }

    if (!existsSync(tokenizerPath)) {
        throw new Error(`Missing tokenizer file at: ${tokenizerPath}`);
    }

    const tokenizer = await loadTokenizer(tokenizerPath);
    const model = await loadModel(modelPath);

    const CSV_DATA_PATH = 'abcnews-date-text_train.csv';
    await FS.rm('index.bin', { force: true });
    await FS.rm('quantized-index.bin', { force: true });

    const numberFormatter = new Intl.NumberFormat();


    await using file = await FS.open(CSV_DATA_PATH, 'r');
    await using fullIndexFile = await FS.open('data/index.bin', 'a+');
    await using quantizedIndexFile = await FS.open('data/quantized-index.bin', 'a+');

    const LIMIT = 1_000_000;
    const LIMIT_STRING = numberFormatter.format(LIMIT);

    const embedding = new Float32Array(256 * LIMIT);
    const quantized = new Int32Array(8 * LIMIT);

    console.time(`creating Embeddings for ${LIMIT_STRING} headlines`);

    let count = -1;
    for await (const line of file.readLines()) {
        if (count >= LIMIT) {
            console.log(`Hit limit at ${count}`);
            break;
        }

        // skip the headers
        if (count < 0) {
            count++;
            continue;
        }

        const headline = line.slice(line.indexOf(',') + 1);
        if (headline.trim() != '') {
            const tokens = tokenize(headline, tokenizer);
            encode(model, tokens, embedding, count);
        }

        count++;
    }


    console.timeEnd(`creating Embeddings for ${LIMIT_STRING} headlines`)
    console.time(`Quantizing ${LIMIT_STRING} embeddings`)
    for (let i = 0; i < count; i++) {
        quantizeTo1Bit(embedding, quantized, i);
    }
    console.timeEnd(`Quantizing ${LIMIT_STRING} embeddings`)
    console.time(`Writing index files`)

    await quantizedIndexFile.write(quantized);
    await fullIndexFile.write(embedding);
    console.timeEnd(`Writing index files`)
}