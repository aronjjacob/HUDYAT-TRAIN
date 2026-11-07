// Usage: node app_predict.js <export dir> <output.json>
// Loads an export the way the HUDYAT app does and predicts its parity.json inputs. buildModel and loadWeights are
// copied from HUDYAT/client/src/features/fsl-recognition/category-core.js; keep them identical to the app's.
// Writes {"pred": [[...]]}.
const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs");

function buildModel(tf, classes, features = 128) {
  const l = tf.layers;
  const model = tf.sequential();
  model.add(l.conv1d({inputShape:[32,features], filters:64, kernelSize:3, padding:'same', activation:'relu'}));
  model.add(l.batchNormalization({momentum:0.99, epsilon:0.0001}));
  model.add(l.maxPooling1d({poolSize:2}));
  model.add(l.dropout({rate:0.3}));
  model.add(l.conv1d({filters:128, kernelSize:3, padding:'same', activation:'relu'}));
  model.add(l.batchNormalization({momentum:0.99, epsilon:0.0001}));
  model.add(l.dropout({rate:0.3}));
  model.add(l.bidirectional({layer:l.lstm({units:128, returnSequences:true,
    activation:'tanh', recurrentActivation:'sigmoid', recurrentDropout:0, dropout:0, kernelInitializer:'zeros', recurrentInitializer:'zeros'}), mergeMode:'concat'}));
  model.add(l.dropout({rate:0.3}));
  model.add(l.bidirectional({layer:l.lstm({units:64, returnSequences:false,
    activation:'tanh', recurrentActivation:'sigmoid', recurrentDropout:0, dropout:0, kernelInitializer:'zeros', recurrentInitializer:'zeros'}), mergeMode:'concat'}));
  model.add(l.dropout({rate:0.3}));
  model.add(l.dense({units:64, activation:'relu'}));
  model.add(l.dense({units:classes, activation:'softmax'}));
  return model;
}

function loadWeights(tf, model, metadata, buffer) {
  // hand-face-location-v1 (Family): the 128 hand values + 13 for where the hands are against the face.
  const features = metadata.preprocessing?.normalization === 'hand-face-location-v1' ? 141 : 128;
  if (metadata.format !== 'fsl-conv1d-bilstm-v1' || metadata.inputShape.join(',') !== '32,' + features)
    throw new Error('This model format is unsupported.');
  const tensors = [];
  try {
    for (const spec of metadata.weights) {
      if (spec.dtype !== 'float32' || spec.offset % 4 || spec.offset + spec.bytes > buffer.byteLength)
        throw new Error('Invalid model weight data.');
      const count = spec.shape.reduce((a,b)=>a*b,1);
      if (count * 4 !== spec.bytes) throw new Error('Invalid weight shape.');
      tensors.push(tf.tensor(new Float32Array(buffer.slice(spec.offset,spec.offset+spec.bytes)),spec.shape));
    }
    let offset=0;
    for(const layer of model.layers){
      const count=layer.weights.length;
      layer.setWeights(tensors.slice(offset,offset+count));offset+=count;
    }
    if(offset!==tensors.length)throw new Error('Unexpected model weight count.');
  } finally { tensors.forEach(t=>t.dispose()); }
}

async function main() {
  const [dir, outFile] = process.argv.slice(2);
  await tf.setBackend("cpu");
  const metadata = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"));
  const bin = fs.readFileSync(path.join(dir, "weights.bin"));
  const buffer = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  const model = buildModel(tf, metadata.classes.length, metadata.inputShape[1]);
  loadWeights(tf, model, metadata, buffer);
  const probe = JSON.parse(fs.readFileSync(path.join(dir, "parity.json"), "utf8"));
  const y = model.predict(tf.tensor(probe.inputs));
  fs.writeFileSync(outFile, JSON.stringify({ pred: await y.array() }));
}
main().catch(e => { console.error(e); process.exit(1); });
