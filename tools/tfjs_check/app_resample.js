// Usage: node app_resample.js <input.json> <output.json>
// input.json: {"clips": [[frame, ...], ...]} with 128-value frames in the app's layout. Writes {"out": [...]}, each
// clip resampled to 32 frames by the HUDYAT app's resample, copied from
// HUDYAT/client/src/features/fsl-recognition/category-core.js; keep it identical to the app's.
const fs = require("fs");

function resample(frames) {
  if (!frames.length || frames.some(f=>f.length!==128)) throw new Error('No valid frames captured.');
  const out = new Float32Array(32*128);
  for (let t=0;t<32;t++) {
    const position=t*(frames.length-1)/31;
    const left=Math.floor(position), right=Math.min(left+1,frames.length-1);
    const ratio=position-left, nearest=Math.floor(position+0.5);
    for(let c=0;c<126;c++) out[t*128+c]=frames[left][c]*(1-ratio)+frames[right][c]*ratio;
    for(let h=0;h<2;h++) {
      out[t*128+126+h]=frames[nearest][126+h];
      if(!out[t*128+126+h]) out.fill(0,t*128+h*63,t*128+(h+1)*63);
    }
  }
  return out;
}

const [inFile, outFile] = process.argv.slice(2);
const clips = JSON.parse(fs.readFileSync(inFile, "utf8")).clips;
const out = clips.map(c => {
  const flat = Array.from(resample(c));
  return Array.from({length: 32}, (_, t) => flat.slice(t * 128, (t + 1) * 128));
});
fs.writeFileSync(outFile, JSON.stringify({ out }));
