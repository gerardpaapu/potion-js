export function normalizeL2(vector: Float32Array, offset = 0, length = 256) {
  let sumOfSquares = 0;
  let end = offset + length;
  for (let i = offset; i < end; i++) {
    let n = vector[i];
    sumOfSquares += n * n;
  }
  let vecLength = Math.sqrt(sumOfSquares);
  for (let i = offset; i < end; i++) {
    vector[i] /= vecLength;
  }
}

export function distanceSquared(a: Float32Array, b: Float32Array, offset = 0) {
  const j = offset * 256; // each embedding is 256 entries
  let d2 = 0;

  for (let i = 0; i < 256; i++) {
    const aa = a[i];
    const bb = b[j + i];
    const diff = aa - bb;
    d2 += diff ** 2;
  }

  return d2;
}
