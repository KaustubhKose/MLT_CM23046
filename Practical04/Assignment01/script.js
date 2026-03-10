let model;

async function loadModel() {
    document.getElementById("result").innerHTML = "Loading MobileNet...";
    model = await mobilenet.load();
    document.getElementById("result").innerHTML = "MobileNet Loaded!";
}

loadModel();

window.classifyImage = async () => {
    const img = document.getElementById("img");
    const pred = await model.classify(img);

    let out = "<b>Top 3 Predictions:</b><br>";
    pred.slice(0, 3).forEach((p, i) => {
        out += `${i + 1}. ${p.className} — ${(p.probability * 100).toFixed(2)}%<br>`;
    });

    document.getElementById("result").innerHTML = out;
};