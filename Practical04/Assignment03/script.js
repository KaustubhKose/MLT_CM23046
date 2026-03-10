let mobilenetModel;
let denseModel;
let index = 1;

async function loadModels() {
    document.getElementById("result").innerHTML = "Loading models...";

    // Load MobileNet
    mobilenetModel = await mobilenet.load();

    // Create simple Dense neural network (untrained)
    denseModel = tf.sequential();
    denseModel.add(tf.layers.flatten({ inputShape: [224, 224, 3] }));
    denseModel.add(tf.layers.dense({ units: 10, activation: "softmax" }));
    denseModel.compile({ optimizer: "sgd", loss: "categoricalCrossentropy" });

    document.getElementById("result").innerHTML = "Models loaded!";
}

loadModels();

// NEXT IMAGE
window.next = () => {
    index = index === 5 ? 1 : index + 1;
    document.getElementById("img").src = `img${index}.jpg`;
    document.getElementById("result").innerHTML = "";
};

// PREVIOUS IMAGE
window.prev = () => {
    index = index === 1 ? 5 : index - 1;
    document.getElementById("img").src = `img${index}.jpg`;
    document.getElementById("result").innerHTML = "";
};

// COMPARE MODELS
window.compare = async () => {
    const img = document.getElementById("img");

    // -------------------------
    // MOBILE NET PREDICTION
    // -------------------------
    const mobilePred = await mobilenetModel.classify(img);

    let output = `<b>MOBILENET PREDICTION:</b><br>`;
    output += `${mobilePred[0].className} — ${(mobilePred[0].probability * 100).toFixed(2)}%<br><br>`;

    // -------------------------
    // DENSE MODEL (UNTRAINED)
    // -------------------------
    output += `<b>DENSE MODEL PREDICTION:</b><br>`;
    output += `Dense model is untrained → random values only.<br><br>`;

    // Academic comparison text
    output += `<b>Comparison:</b><br>
    • MobileNet is pre-trained → accurate<br>
    • Dense model is untrained → inaccurate<br>
    • Shows importance of transfer learning`;

    document.getElementById("result").innerHTML = output;
};