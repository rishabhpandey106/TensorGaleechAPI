import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import csv from "csv-parser";
import { Transform } from "stream";

interface Row {
  embedding: string; // The embedding will now be a valid JSON string
  rating: number;    // The rating (0 or 1) for profanity
}

function createLineRangeStream(startLine: number, endLine: number) {
  let currentLine = 0;
  return new Transform({
    transform(chunk, _, callback) {
      if (currentLine >= startLine && currentLine < endLine) {
        this.push(chunk);
      }
      currentLine++;
      if (currentLine >= endLine) {
        this.push(null);
      }
      callback();
    },
    objectMode: true,
  });
}

async function parseCSV(
  filePath: string,
  startLine: number,
  endLine: number
): Promise<Row[]> {
  return new Promise((resolve, reject) => {
    const rows: Row[] = [];

    fs.createReadStream(filePath)
      .pipe(csv({ separator: "," })) // Adjust CSV delimiter
      .pipe(createLineRangeStream(startLine, endLine))
      .on("data", (row) => {
        rows.push(row);
      })
      .on("error", (error) => {
        reject(error);
      })
      .on("end", () => {
        resolve(rows);
      });
  });
}

class AI {
  compile() {
    const model = tf.sequential();

    // Input layer (adjust for the size of your embeddings)
    model.add(
      tf.layers.dense({
        units: 3,
        inputShape: [384], // Adjust this to match the size of your embeddings
      })
    );

    // Output layer
    model.add(
      tf.layers.dense({
        units: 1,
        activation: "sigmoid",
      })
    );

    model.compile({
      loss: "binaryCrossentropy",
      optimizer: "sgd",
      metrics: ["accuracy"],
    });

    return model;
  }

  async run() {
    const model = this.compile();

    const data = await parseCSV("prepared_dataset.csv", 0, 45000);

    const converted = data.map((row) => ({
      embedding: JSON.parse(row.embedding), // Embedding should now be parsed correctly
      rating: Number(row.rating),
    }));

    const xsConverted = converted.map(({ embedding }) => embedding);
    const ysConverted = converted.map(({ rating }) => [rating]);

    const xs = tf.tensor2d(xsConverted);
    const ys = tf.tensor2d(ysConverted);

    await model.fit(xs, ys, {
      epochs: 250,
    });

    // Save the model after training
    await model.save("file://./profanity-model");
  }
}

const ai = new AI();
ai.run();
