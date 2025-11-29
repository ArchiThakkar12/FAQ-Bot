import { pipeline } from "@xenova/transformers";

let rephraserPipe;

export async function initRephraser() {
  rephraserPipe = await pipeline(
    "text2text-generation",
    "Xenova/flan-t5-small"
  );
}

export async function rephraseQuestion(userInput) {
  const prompt =
    `Rewrite this question so that it is clean, clear, and easy for a machine to understand:\nUser: ${userInput}\nRewritten:`;

  const output = await rephraserPipe(prompt, {
    max_new_tokens: 40,
  });

  return output[0].generated_text.split("Rewritten:")[1]?.trim();
}