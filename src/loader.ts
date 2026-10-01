import * as FS from "node:fs/promises";

const HEADER_SIZE_BYTES = 8;

export interface PotionSections {
    weights: SectionLocation;
    mapping: SectionLocation;
    embeddings: SectionLocation;
}

export interface PotionManifest {
    sections: PotionSections,
    dataStart: number;
}

export type SectionLocation = { dtype: 'F32' | 'I32' | 'I8', shape: number[], data_offsets: [number, number] }

export type PotionModel = {
    weights: Float32Array<ArrayBufferLike>;
    mapping: Uint32Array<ArrayBufferLike>;
    embeddings: Int8Array<ArrayBufferLike>;
}

async function readManifest(fd: FS.FileHandle) {
    // The header looks like it might be 2 32 bit unsigned integers
    // and I think the first one is the length of the manifest
    // no idea what the second one is for, in our model it's 0
    const header = new Uint32Array(2);
    await fd.read(header, 0, HEADER_SIZE_BYTES, 0);

    const sectionsLenBytes = header[0];
    const sectionsRaw = new Uint8ClampedArray(sectionsLenBytes);

    // now starting reading right after the header
    await fd.read(sectionsRaw, 0, sectionsLenBytes, HEADER_SIZE_BYTES);

    // JSON is (by standard) utf-8 at rest, indistiguishable from ascii/latin-1
    // in this case.
    const sectionsString = Buffer.from(sectionsRaw).toString("utf8");
    const sections = JSON.parse(sectionsString) as PotionSections;
    const dataStart = HEADER_SIZE_BYTES + sectionsLenBytes;

    return { sections, dataStart } as PotionManifest;
}

type ArrayOfDtype<T extends 'F32' | 'I32' | 'I8'> =
    T extends 'F32' ? Float32Array
    : T extends 'I32' ? Uint32Array
    : Int8Array;

async function readSection<T extends 'F32' | 'I32' | 'I8'>(
    file: FS.FileHandle, manifest: PotionManifest, dtype: T, sectionName: keyof PotionSections
) {
    const location = manifest.sections[sectionName];
    const { dataStart } = manifest;
    if (dtype !== location.dtype) {
        throw new Error();
    }

    let size = 1;
    for (const len of location.shape) {
        size *= len;
    }

    let data: Int8Array | Uint32Array | Float32Array;
    switch (dtype) {
        case 'I32':
            data = new Uint32Array(size);
            break;
        case 'F32':
            data = new Float32Array(size);
            break;
        case 'I8':
            data = new Int8Array(size);
    }

    const [start, end] = location.data_offsets;
    const byteLength = end - start;

    await file.read(data, 0, byteLength, start + dataStart);
    return data as ArrayOfDtype<T>;
}

export async function loadModel(path: string) {
    const safetensors = await FS.open(path);
    try {
        const manifest = await readManifest(safetensors);
        const weights = await readSection(safetensors, manifest, 'F32', 'weights');
        const mapping = await readSection(safetensors, manifest, 'I32', 'mapping');
        const embeddings = await readSection(safetensors, manifest, 'I8', 'embeddings');

        return { weights, mapping, embeddings } as PotionModel;
    } finally {
        safetensors.close();
    }
}
