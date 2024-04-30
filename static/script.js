document.addEventListener("DOMContentLoaded", function() {
    setInterval(function() {
        fetch('/get-incidents')  // Replace this with your actual endpoint
            .then(response => response.json())
            .then(data => {
                if (data.length > 0) {
                    const incident = data[0];  // Assuming only one incident for simplicity
                    displayNotification(`Incident (${incident.type}) detected at ${new Date(incident.timestamp)}`);
                }
            })
            .catch(error => console.error('Error fetching incidents:', error));
    }, 5000);  // Poll every 5 seconds
});

function displayNotification(message) {
    // Create and display a popup notification with alert sound
    // Implementation depends on your frontend framework or library (e.g., Bootstrap modal, SweetAlert)
}
