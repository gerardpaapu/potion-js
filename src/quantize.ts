
export function quantizeTo1Bit(embedding: Float32Array, output: Int32Array, offset = 0) {
    let embeddingOffset = offset * 256
    let quantizedOffset = offset * 8;

    for (let i = 0; i < 8; i++) {
        output[i + quantizedOffset] = 0;
        for (let j = 0; j < 32; j++) {
            output[i + quantizedOffset] |= (embedding[embeddingOffset + j + 32 * i] > 0 ? 1 : 0) << j;
        }
    }

    return output;
}

export function popcount32(x: number): number {
    x -= (x >>> 1) & 0x55555555;
    x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
    return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

export function hammingDistance256(a: Int32Array, b: Int32Array, offset = 0): number {
    return (
        popcount32(a[0] ^ b[0 + offset]) +
        popcount32(a[1] ^ b[1 + offset]) +
        popcount32(a[2] ^ b[2 + offset]) +
        popcount32(a[3] ^ b[3 + offset]) +
        popcount32(a[4] ^ b[4 + offset]) +
        popcount32(a[5] ^ b[5 + offset]) +
        popcount32(a[6] ^ b[6 + offset]) +
        popcount32(a[7] ^ b[7 + offset])
    );
}