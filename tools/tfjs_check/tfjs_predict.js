// Usage: node tfjs_predict.js <tfjs dir> <input.json> <output.json>
// input.json: {"x": [[[...128]...32]...N]}. Writes {"pred": [[...]], "weights": [names]}.
const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs");

async function main() {
  const [dir, inFile, outFile] = process.argv.slice(2);
  await tf.setBackend("cpu");
  const mj = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"));
  const specs = [], buffers = [];
  for (const g of mj.weightsManifest) {
    specs.push(...g.weights);
    for (const p of g.paths) buffers.push(fs.readFileSync(path.join(dir, p)));
  }
  const all = Buffer.concat(buffers);
  const weightData = all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength);
  const model = await tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: mj.modelTopology, weightSpecs: specs, weightData, format: mj.format,
  }), { strict: true });
  const x = JSON.parse(fs.readFileSync(inFile, "utf8")).x;
  const y = model.predict(tf.tensor3d(x));
  fs.writeFileSync(outFile, JSON.stringify({
    pred: await y.array(), weights: model.weights.map(w => w.originalName || w.name),
  }));
}
main().catch(e => { console.error(e); process.exit(1); });
