export function normalizeL2(vector: Float32Array) {
    let sumOfSquares = 0;
    for (let i = 0; i < vector.length; i++) {
        let n = vector[i];
        sumOfSquares += n * n;
    }
    let vecLength = Math.sqrt(sumOfSquares);
    for (let i = 0; i < vector.length; i++) {
        vector[i] /= vecLength;
    }
}


export function cosineDistance(
    a: Float32Array,
    b: Float32Array
) {
    let d2 = 0;
    for (let i = 0; i < a.length; i++) {
        let aa = a[i];
        let bb = b[i];
        let diff = aa - bb;

        d2 += diff ** 2;
    }

    return Math.sqrt(d2);
}

