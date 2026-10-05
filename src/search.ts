import * as Path from "node:path/posix";
import { existsSync } from "node:fs";
import { loadTokenizer, tokenize } from "./tokenizer.ts";
import { loadModel } from "./loader.ts";
import { encode } from "./encoder.ts";
import { hammingDistance256, quantizeTo1Bit } from "./quantize.ts";

import * as FS from "node:fs/promises";
import * as S from "node:stream/consumers";
import { distanceSquared } from "./vector.ts";

export async function main() {
  const MODELS_PATH =
    process.env.MODELS_PATH ??
    Path.join(process.env.HOME!, ".local/share/models");
  const potionPath = Path.join(MODELS_PATH, "potion-mxbai-micro");
  const modelPath = Path.join(potionPath, "model.safetensors");
  const tokenizerPath = Path.join(potionPath, "tokenizer.json");

  if (!existsSync(modelPath)) {
    throw new Error(`Missing model file at: ${modelPath}`);
  }

  if (!existsSync(tokenizerPath)) {
    throw new Error(`Missing tokenizer file at: ${tokenizerPath}`);
  }

  const tokenizer = await loadTokenizer(tokenizerPath);
  const model = await loadModel(modelPath);
  const needle = await S.text(process.stdin);
  const tokens = tokenize(needle.toLowerCase(), tokenizer);

  const needleEmbedding = new Float32Array(256);
  const needleEmbeddingQ1 = new Int32Array(8);

  encode(model, tokens, needleEmbedding);
  quantizeTo1Bit(needleEmbedding, needleEmbeddingQ1);

  const FAST_INDEX_ONLY = process.argv.slice(1).includes("--fast-index-only");
  const K = FAST_INDEX_ONLY ? 10 : 100;
  const N = 10;

  // We're going to keep the K best (shortest distance) candidates from
  // this first pass using cheap hamming distance as an approximation.
  //
  // Even though the scores won't ge as accurate, it's very likely that the
  // true 10 best are within the 100 approximate best candidates
  let firstPassResults = mkResults(K);
  {
    await using qIndex = await FS.open("./data/quantized-index.bin", "r");
    const QUANTIZED_INDEX_LENGTH = 8 * 1_000_000;
    const QUANTIZED_INDEX_LENGTH_BYTES = QUANTIZED_INDEX_LENGTH * 4;
    const qIndexMemory = new Int32Array(QUANTIZED_INDEX_LENGTH);

    // read the whole goddamn file
    await qIndex.read(qIndexMemory, 0, QUANTIZED_INDEX_LENGTH_BYTES);

    for (let i = 0; i < 1_000_000; i++) {
      const distance = hammingDistance256(
        needleEmbeddingQ1,
        qIndexMemory,
        i * 8,
      );
      const candidate = { distance, location: i };
      insertCandidateSorted(candidate, firstPassResults);
    }
  }

  let bestResults;
  if (FAST_INDEX_ONLY) {
    bestResults = firstPassResults;
  } else {
    bestResults = mkResults(N);
    await using index = await FS.open("./data/index.bin", "r");
    const INDEX_LENGTH = 256 * 1_000_000;
    const INDEX_LENGTH_BYTES = INDEX_LENGTH * 4;

    // again, we're just going to read the whole file into memory
    const indexMemory = new Float32Array(INDEX_LENGTH);
    await index.read(indexMemory, 0, INDEX_LENGTH_BYTES);

    // this time we're looping through the top-K candidates from the first
    // pass, and keeping the top 10 by the cosine metric
    for (let i = 0; i < K; i++) {
      const { location } = firstPassResults[i];
      const distance = distanceSquared(needleEmbedding, indexMemory, location);

      insertCandidateSorted({ distance, location }, bestResults);
    }
  }

  // this probably isn't the fastest way to do this;
  const indexes = bestResults.map((obj) => obj.location).toReversed();
  const results = [];

  const CSV_DATA_PATH = "data/abcnews-date-text_train.csv";
  await using file = await FS.open(CSV_DATA_PATH, "r");

  let lineNum = -1;
  let found = 0;
  for await (const line of file.readLines()) {
    let idx = indexes.indexOf(lineNum);
    if (idx !== -1) {
      found++;
      let headline = line.slice(line.indexOf(",") + 1);
      let score = bestResults[idx].distance;
      results[idx] = { score, headline };
    }

    if (found >= K) {
      console.log(`Found ${K} items`);
      break;
    }

    lineNum++;
  }

  for (let i = 0; i < results.length; i++) {
    const result = results[i];
    console.log(
      `${(i + 1).toString().padStart(2, " ")}. (${result.score.toFixed(2)}) ${result.headline.charAt(0).toLocaleUpperCase()}${result.headline.slice(1)}`,
    );
  }
}

type Candidate = { distance: number; location: number };

function mkResults(length: number) {
  const results = [] as Candidate[];
  const BAD_RESULT = { distance: Infinity, location: -1 } as const;
  while (results.length < length) {
    results.push(BAD_RESULT);
  }

  return results;
}

function insertCandidateSorted(candidate: Candidate, top: Candidate[]) {
  const { distance } = candidate;
  if (distance < top[0].distance) {
    // find where we want it
    let target = 0;
    while (target < top.length && top[target].distance > distance) {
      target++;
    }
    // this should be the last position where we were closer
    target--;

    // move everything starting at that position down one
    for (let i = 0; i < target; i++) {
      top[i] = top[i + 1];
    }

    // insert our new value (sorted)
    top[target] = candidate;
  }
}
