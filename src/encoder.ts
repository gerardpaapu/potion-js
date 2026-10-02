import type { PotionModel } from "./loader.ts";
import { normalizeL2 } from "./vector.ts";

export function encode(model: PotionModel, tokens: number[], output?: Float32Array, offset = 0) {
    const { weights, mapping, embeddings } = model;

    const tokensLength = tokens.length;
    // Our resulting embeddings are exactly 256 32 bit floats aka 1024 bytes
    const EMBEDDING_SIZE = 256;
    const start = offset * EMBEDDING_SIZE;

    const embedding = output ?? new Float32Array(EMBEDDING_SIZE);
    for (let i = 0; i < tokensLength; i++) {
        const id = tokens[i];
        const weight = weights[id];
        const src = mapping[id] * EMBEDDING_SIZE;

        for (let j = 0; j < EMBEDDING_SIZE; j++) {
            // Add the sum of each token's embedding into 
            // each column of the embedding
            embedding[j + start] += embeddings[src + j] * weight;
        }
    }

    for (let i = 0; i < EMBEDDING_SIZE; i++) {
        // convert the sum to a mean
        embedding[i + start] /= tokensLength;
    }

    // L2 aka. Euclidian norm
    normalizeL2(embedding, start, EMBEDDING_SIZE);

    return embedding;
}