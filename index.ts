import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import csv from "csv-parser";
import { Transform } from "stream";

interface Row {
  embedding: string; 
  rating: number;    
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
      .pipe(csv({ separator: "," })) 
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
  model: tf.Sequential;

  compile() {
    const model = tf.sequential();

    // input layer
    model.add(
      tf.layers.dense({
        units: 3,
        inputShape: [768], // adjust this to match the size of embeddings
      })
    );

    // output layer
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

  constructor() {
    this.model = this.compile();
  }

  async evaluateMetrics(xs: tf.Tensor, ys: tf.Tensor) {
    const predictions = this.model.predict(xs) as tf.Tensor;

    // Round the predictions to 0 or 1
    const roundedPredictions = predictions.round();

    const truePositives = tf.sum(ys.mul(roundedPredictions));
    const falsePositives = tf.sum(ys.mul(tf.scalar(1).sub(roundedPredictions)));
    const falseNegatives = tf.sum(tf.scalar(1).sub(ys).mul(roundedPredictions));
    const trueNegatives = tf.sum(tf.scalar(1).sub(ys).mul(tf.scalar(1).sub(roundedPredictions)));

    // Calculate Precision, Recall, F1 score
    const precision = truePositives.div(truePositives.add(falsePositives));
    const recall = truePositives.div(truePositives.add(falseNegatives));
    const f1Score = precision.mul(recall).mul(2).div(precision.add(recall));

    return { precision, recall, f1Score };
  }

  async run() {
    const model = this.compile();

    const data = await parseCSV("prepared_dataset_cleaned.csv", 0, 45000);

    const converted = data.map((row) => ({
      embedding: JSON.parse(row.embedding),
      rating: Number(row.rating),
    }));

    const xsConverted = converted.map(({ embedding }) => embedding);
    const ysConverted = converted.map(({ rating }) => [rating]);

    const xs = tf.tensor2d(xsConverted);
    const ys = tf.tensor2d(ysConverted);

    // Train the model
    await model.fit(xs, ys, {
      epochs: 250,
    });

    await model.save("file://./profanity-model-v5");

    const metrics = await this.evaluateMetrics(xs, ys);

    console.log(`Precision: ${metrics.precision}`);
    console.log(`Recall: ${metrics.recall}`);
    console.log(`F1 Score: ${metrics.f1Score}`);
  }
}

const ai = new AI();
ai.run();
