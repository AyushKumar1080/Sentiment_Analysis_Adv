"use strict";

const sentimentForm = document.getElementById("sentimentForm");
const reviewInput = document.getElementById("review");
const status = document.getElementById("status");
const results = document.getElementById("results");
const resultsBody = document.getElementById("resultsBody");

sentimentForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const review = reviewInput.value.trim();

    if (!review) {
        status.textContent = "Please enter a product review.";
        return;
    }

    const button = sentimentForm.querySelector("button");
    button.disabled = true;

    status.textContent = "Analyzing sentiment...";
    results.hidden = true;
    resultsBody.innerHTML = "";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                review: review
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Analysis failed.");
        }

        const predictions = data.results || data.predictions || [];

        if (predictions.length === 0) {
            status.textContent = "No aspects were found.";
            return;
        }

        predictions.forEach(function (item) {
            const row = document.createElement("tr");

            const aspectCell = document.createElement("td");
            aspectCell.textContent = item.aspect || "Unknown";

            const sentimentCell = document.createElement("td");
            const sentiment = item.sentiment || "Unknown";
            sentimentCell.textContent = sentiment;

            sentimentCell.className = sentiment.toLowerCase();

            const confidenceCell = document.createElement("td");
            const confidence = item.confidence;

            if (typeof confidence === "number") {
                confidenceCell.textContent =
                    (confidence <= 1
                        ? confidence * 100
                        : confidence
                    ).toFixed(2) + "%";
            } else {
                confidenceCell.textContent = "N/A";
            }

            row.appendChild(aspectCell);
            row.appendChild(sentimentCell);
            row.appendChild(confidenceCell);

            resultsBody.appendChild(row);
        });

        results.hidden = false;
        status.textContent = "Analysis completed successfully.";

    } catch (error) {
        status.textContent = "Error: " + error.message;
        console.error("Sentiment analysis error:", error);

    } finally {
        button.disabled = false;
    }
});
