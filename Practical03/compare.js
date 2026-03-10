let denseModel;
let lstmModel;
let tokenizer = {};
let vocabSize = 0;
let maxLen = 6;

const sentences = [
"I love this product",
"This is amazing",
"Fantastic service",
"I hate this",
"Very bad experience",
"Terrible quality",
"Really good",
"Not good",
"I enjoy this",
"Awful item"
];

const labels = [1,1,1,0,0,0,1,0,1,0];

function tokenize(text){
return text.toLowerCase().split(" ");
}

function buildVocab(){
let index = 1;
sentences.forEach(s=>{
tokenize(s).forEach(word=>{
if(!tokenizer[word]){
tokenizer[word]=index++;
}
});
});
vocabSize=index;
}

function encode(sentence){
let tokens=tokenize(sentence);
let arr=new Array(maxLen).fill(0);

tokens.slice(0,maxLen).forEach((t,i)=>{
arr[i]=tokenizer[t]||0;
});

return arr;
}

function prepareData(){

buildVocab();

let xs=sentences.map(s=>encode(s));
let ys=labels;

return {
x:tf.tensor2d(xs),
y:tf.tensor2d(ys,[ys.length,1])
};

}

function buildDense(){

const model=tf.sequential();

model.add(tf.layers.embedding({
inputDim:vocabSize,
outputDim:16,
inputLength:maxLen
}));

model.add(tf.layers.globalAveragePooling1d());

model.add(tf.layers.dense({units:16,activation:'relu'}));

model.add(tf.layers.dense({units:1,activation:'sigmoid'}));

model.compile({
optimizer:'adam',
loss:'binaryCrossentropy',
metrics:['accuracy']
});

return model;

}

function buildLSTM(){

const model=tf.sequential();

model.add(tf.layers.embedding({
inputDim:vocabSize,
outputDim:16,
inputLength:maxLen
}));

model.add(tf.layers.lstm({units:16}));

model.add(tf.layers.dense({units:1,activation:'sigmoid'}));

model.compile({
optimizer:'adam',
loss:'binaryCrossentropy',
metrics:['accuracy']
});

return model;

}

async function trainBoth(){

const data=prepareData();

denseModel=buildDense();
lstmModel=buildLSTM();

const denseHist=await denseModel.fit(data.x,data.y,{
epochs:30,
verbose:0
});

const lstmHist=await lstmModel.fit(data.x,data.y,{
epochs:30,
verbose:0
});

document.getElementById("denseAcc").innerText=
denseHist.history.acc?.slice(-1)[0]?.toFixed(3) ||
denseHist.history.accuracy.slice(-1)[0].toFixed(3);

document.getElementById("denseLoss").innerText=
denseHist.history.loss.slice(-1)[0].toFixed(3);

document.getElementById("lstmAcc").innerText=
lstmHist.history.acc?.slice(-1)[0]?.toFixed(3) ||
lstmHist.history.accuracy.slice(-1)[0].toFixed(3);

document.getElementById("lstmLoss").innerText=
lstmHist.history.loss.slice(-1)[0].toFixed(3);

alert("Training complete");

}

async function predictBoth(){

const text=document.getElementById("inputText").value;

const encoded=tf.tensor2d([encode(text)]);

const densePred=await denseModel.predict(encoded).data();
const lstmPred=await lstmModel.predict(encoded).data();

showResult("dense",densePred[0]);
showResult("lstm",lstmPred[0]);

}

function showResult(type,val){

let verdict=val>0.5?"Positive":"Negative";
let cls=val>0.5?"pos":"neg";

document.getElementById(type+"Verdict").innerText=verdict;
document.getElementById(type+"Verdict").className="verdict "+cls;
document.getElementById(type+"Confidence").innerText=
"Confidence: "+(val*100).toFixed(1)+"%";

}

function resetAll(){
location.reload();
}
