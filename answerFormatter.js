import { pipeline } from "@xenova/transformers";

let formatter;

export async function initFormatter() {
  formatter = await pipeline(
    "text-generation",
    "Xenova/Qwen2.5-0.5B-Instruct"
  );
}

export async function formatAnswer(userQuestion, rawAnswer) {
  const prompt =
    `User asked: "${userQuestion}"\n` +
    `Answer in a natural, friendly, clear tone:\n`;

  const response = await formatter(prompt + rawAnswer, {
    max_new_tokens: 80,
  });

  return response[0].generated_text.split("tone:")?.[1]?.trim() || rawAnswer;
}
