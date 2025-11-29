import readline from "readline";
import { initRephraser } from "./rephraser.js";
import { initRetriever } from "./retriever.js";
import { initFormatter } from "./answerFormatter.js";
import { processQuestion } from "./bot.js";

await initRephraser();
await initRetriever();
await initFormatter();

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

console.log("FAQ Bot Ready!");

rl.on("line", async (input) => {
  const response = await processQuestion(input);

  if (response.type === "clarification") {
    console.log(response.message);
    console.log("Options:");
    response.options.forEach((opt, i) =>
      console.log(`${i + 1}. ${opt}`)
    );
  }

  if (response.type === "answer") {
    console.log("\n→ " + response.answer + "\n");
  }
});
