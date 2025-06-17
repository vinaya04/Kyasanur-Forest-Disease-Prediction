document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('predictionForm');

    // Define the correct feature order (must match selected_features in Python)
    const featureKeys = [
        "sudden_chills", "headache", "severe_myalgia", "fever",
        "neck_stiffness", "unconsciousness", "low_bp",
        "Season_P", "GPS_loc_P", "Occupation_P"
    ];

    form.addEventListener('submit', async function (event) {
        event.preventDefault();  // Prevent form from submitting normally

        // Gather the form data and ensure it's in the correct order
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });

        // Create a feature array in the correct order
        const features = featureKeys.map(key => parseInt(data[key]));

        // Send the data to the server (use your server endpoint for prediction)
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: JSON.stringify({ features }),  // Send the features array
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            // Check if the response is successful
            if (response.ok) {
                const result = await response.json();  // Parse the JSON response
                const resultDiv = document.getElementById('predictionResult');
                resultDiv.style.display = 'block';
                resultDiv.innerText = `Prediction: ${result.class_label} (Accuracy: ${result.model_accuracy})`;

                // Show an alert based on the prediction result
                if (result.class_label.includes('Confirmed')) {
                    alert('KFD detected. Please consult a medical professional immediately.');
                } else {
                    alert('No signs of KFD detected. Stay safe and take precautions.');
                }
            } else {
                alert("Prediction failed. Please try again!");
            }
        } catch (error) {
            console.error("Error:", error);
            alert("An error occurred while making the prediction. Please try again.");
        }
    });
});

