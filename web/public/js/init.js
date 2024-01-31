function selectDataset(dataset) {
    // Update the selected dataset
    selectedDataset = dataset;
    // // Update the button label
    // document.getElementById('dataset-button').textContent = dataset;
    document.getElementById('selected-dataset').value = selectedDataset; // Update hidden input field

    // Highlight the selected dataset
    updateDatasetHighlighting(dataset);
}

function updateDatasetHighlighting(selectedDataset) {
    // Remove highlighting from all dataset options
    document.querySelectorAll('.dropdown-content a').forEach(function(link) {
        link.classList.remove('selected-object');
    });

    // Add highlighting to the selected dataset
    let selectedLink = document.querySelector('.dropdown-content a[onclick*="' + selectedDataset + '"]');
    if (selectedLink) {
        selectedLink.classList.add('selected-object');
    }
}

// Initialize highlighting on page load
document.addEventListener('DOMContentLoaded', function() {
    updateDatasetHighlighting(selectedDataset);
});