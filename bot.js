import { rephraseQuestion } from "./rephraser.js";
import { findRelevantFAQ } from "./retriever.js";
import { formatAnswer } from "./answerFormatter.js";
import fs from "fs";

export async function processQuestion(userInput) {

  // Step 1: Rephrase
  const cleanQ = await rephraseQuestion(userInput);

  // Step 2: Retrieve
  const { maxScore, bestIndex, matches } = await findRelevantFAQ(cleanQ);
  const faqs = JSON.parse(fs.readFileSync("faqs.json"));

  if (matches.length > 1 && maxScore < 0.80) {
    const options = matches.map(m => faqs[m.idx].question);
    return {
      type: "clarification",
      message: "I found multiple related questions. Which one do you mean?",
      options
    };
  }

  // Step 3: Format final answer
  const final = await formatAnswer(userInput, faqs[bestIndex].answer);

  return {
    type: "answer",
    answer: final
  };
}
