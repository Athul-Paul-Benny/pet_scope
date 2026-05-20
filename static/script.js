const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const resultsDiv = document.getElementById("results");

let selectedFile = null;

// PREVIEW
fileInput.addEventListener("change", () => {
    selectedFile = fileInput.files[0];

    if (selectedFile) {
        preview.src = URL.createObjectURL(selectedFile);
        preview.style.display = "block";
    }
});

// PREDICT
async function predict() {

    if (!selectedFile) {
        alert("Upload an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    resultsDiv.innerHTML = `<div class="loader"></div><p>Analyzing...</p>`;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        const top = data.predictions[0];

        resultsDiv.innerHTML = `
            <div class="result-card">
                <h2>${top.breed}</h2>
                <p>Confidence: ${top.confidence.toFixed(2)}%</p>
            </div>
        `;

    } catch (err) {
        resultsDiv.innerHTML = "❌ Error occurred.";
    }
}