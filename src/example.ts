import * as Path from 'node:path/posix';
import { existsSync } from 'node:fs';
import { loadTokenizer, tokenize } from './tokenizer.ts'
import { loadModel } from './loader.ts';
import { encode } from './encoder.ts';

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

    const tokens = tokenize("just like clockwork", tokenizer);
    console.log(`tokens = ${tokens}`);

    const embedding = encode(model, tokens);

    console.log([...embedding]);
}