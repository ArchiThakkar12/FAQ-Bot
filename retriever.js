import fs from "fs";
import { pipeline } from "@xenova/transformers";

let embedder;
let faqs = [];
let faqVectors = [];

export async function initRetriever() {
  faqs = JSON.parse(fs.readFileSync("faqs.json", "utf8"));

  embedder = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );

  faqVectors = await Promise.all(
    faqs.map(f => embedder(f.question, {
      pooling: "mean",
      normalize: true
    }).then(r => r.data))
  );
}

function cosine(a, b) {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

export async function findRelevantFAQ(query) {
  const qVec = (await embedder(query, { pooling: "mean", normalize: true })).data;

  let scores = faqVectors.map(v => cosine(qVec, v));

  let maxScore = Math.max(...scores);
  let bestIndex = scores.indexOf(maxScore);

  // multiple similar options?
  let closeMatches = scores
    .map((score, idx) => ({ score, idx }))
    .filter(x => x.score > 0.55);

  return { maxScore, bestIndex, matches: closeMatches };
}
