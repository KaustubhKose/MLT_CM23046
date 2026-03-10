let model;
let index = 1;

// Load MobileNet
async function loadModel() {
    document.getElementById("result").innerHTML = "Loading MobileNet...";
    model = await mobilenet.load();
    document.getElementById("result").innerHTML = "Model Loaded!";
}
loadModel();

// NEXT IMAGE
window.next = () => {
    index = (index === 5) ? 1 : index + 1;
    document.getElementById("img").src = `img${index}.jpg`;
    document.getElementById("result").innerHTML = "";
};

// PREVIOUS IMAGE
window.prev = () => {
    index = (index === 1) ? 5 : index - 1;
    document.getElementById("img").src = `img${index}.jpg`;
    document.getElementById("result").innerHTML = "";
};

// CLASSIFY
window.classifyImage = async () => {
    const img = document.getElementById("img");

    const predictions = await model.classify(img);

    let txt = `<b>Predictions for Image ${index}:</b><br><br>`;
    predictions.slice(0, 3).forEach((p, i) => {
        txt += `${i + 1}. ${p.className} — ${(p.probability * 100).toFixed(2)}%<br>`;
    });

    document.getElementById("result").innerHTML = txt;
};